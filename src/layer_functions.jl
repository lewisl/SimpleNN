# ============================
# abstract types for layer and normparams
# ============================

using LoopVectorization

# ==========================================
# Layer functions for feedforward and backpropagation
# ==========================================
# feedforward layers are all called layer_forward! and dispatch on the layer type
# backprop layers are all called layer_backward! and dispatch on the layer type
# both methods are supplied for each kind of data_layer: ConvLayer, LinearLayer,
#    FlattenLayer, MaxPoolLayer


# ConvLayer
# functor-style call for feedforward

"""
    (layer::ConvLayer)(x::AbstractArray{ELT, 4}, current_batch_size::Int)

Perform the forward pass of a convolutional layer on a 4D input tensor.

# Arguments
- `layer::ConvLayer`: The convolutional layer containing weights, biases, and other parameters which will
dispatch to this function, so the argument is both function call and argument.
- `x::AbstractArray{ELT, 4}`: Input tensor with dimensions (height, width, channels, batch_size)
- `current_batch_size::Int`: Size of batch, which must be equal to or smaller than the batch_sized used to define layers

# Returns
Nothing. Results are stored in-place in `layer.a`.

# Note
This implementation uses `@turbo` from LoopVectorization.jl for performance optimization.
The tensor dimensions follow Julia's column-major convention: (height, width, channels, batch).
"""
function (layer::ConvLayer)(x::AbstractArray{ELT,4}, current_batch_size::Int)
    cb = current_batch_size
    mb = size(layer.z, 4)
    cb_rng = 1:cb
    if mb == cb
        a_below = layer.a_below
        pad_x = layer.pad_x
        z = layer.z
    else
        a_below = view(layer.a_below, :, :, :, cb_rng)
        x = view(x, :, :, :, cb_rng)  # don't need a non-view of x because it's an input we don't pull from the layer variable
        pad_x = view(layer.pad_x, :, :, :, cb_rng)
        z = view(layer.z, :, :, :, cb_rng)
    end

    a_below .= x
    pad = layer.pad

    if layer.padrule == :same
        @turbo @views pad_x[begin+pad:end-pad, begin+pad:end-pad, :, :] .= x
        use_x = pad_x  # use_x is a "selector" alias
    else
        use_x = x
    end

    # initialize output z with 0.0f0 if nobias or with bias: when dobias==false, layer.bias remains as initialized to zeros
    if layer.dobias
        for oc in axes(layer.z, 3)
            @turbo z[:, :, oc, :] .= layer.bias[oc]
        end
    else
        @turbo fill!(z, ELT(0.0))  # all initialized to zero, 1/2x time of inserting bias    layer.z
    end

    # Combine bias initialization with convolution in a single pass
    @turbo for b in cb_rng  # axes(layer.z, 4)  #  mb_rng  # batch
        for oc in axes(layer.z, 3)     #   # output channels
            for j in axes(layer.z, 2)  # width
                for i in axes(layer.z, 1)  # height
                    for ic in axes(layer.weight, 3)  # input channels
                        for fh in axes(layer.weight, 1)  # filter height
                            for fw in axes(layer.weight, 2)  # filter width
                                z[i, j, oc, b] += use_x[i+fh-1, j+fw-1, ic, b] *   #   layer.z   layer.pad_x
                                                  layer.weight[fh, fw, ic, oc]
                            end
                        end
                    end
                end
            end
        end
    end

    layer.normalizationf(layer, current_batch_size) # either batchnorm! or noop

    # activation
    layer.activationf(layer, current_batch_size)

    return
end

# functor-style call for back propagation
"""
    (layer::ConvLayer)(layer_above, current_batch_size::Int)

Perform the backward pass (backpropagation) for a convolutional layer.

# Arguments
- `layer::ConvLayer`: The convolutional layer containing weights, gradients, and other parameters, so the argument is both function call and argument.
- `layer_above`: The layer above in the network, containing error gradients
- `current_batch_size::Int`: Size of batch, which must be equal to or smaller than the batch_sized used to define layers


# Returns
Nothing. Gradients are computed and stored in-place in `layer.grad_weight`,
`layer.grad_bias` (if `layer.dobias` is true), and `layer.eps_l` (for propagating
errors to lower layers).

# Note
This implementation applies the chain rule through activation functions, normalization,
and the convolution operation to compute gradients for all parameters.
"""
function (layer::ConvLayer)(layer_above, current_batch_size::Int)
    cb = current_batch_size
    cb_rng = 1:cb
    f_h, f_w, ic, oc = size(layer.weight)
    H_out, W_out, _, mb = size(layer.eps_l)
    if mb == cb
        eps_l = layer.eps_l
        grad_a = layer.grad_a
        pad_above_eps = layer.pad_above_eps
        eps_l_above = layer_above.eps_l
    else
        eps_l = view(layer.eps_l, :, :, :, cb_rng)
        grad_a = view(layer.grad_a, :, :, :, cb_rng)
        pad_above_eps = view(layer.pad_above_eps, :, :, :, cb_rng)
        eps_l_above = view(layer_above.eps_l, :, :, :, cb_rng)
    end

    fill!(layer.grad_weight, ELT(0.0))  # reinitialization to allow accumulation of convolutions
    fill!(layer.eps_l, ELT(0.0))
    fill!(layer.grad_a, ELT(0.0))
    inverse_n_samples = ELT(1.0) / ELT(cb)

    layer.activation_gradf(layer, current_batch_size)

    eps_l_above .*= grad_a

    pad = layer.pad

    # @show size(layer.pad_above_eps)
    # @show size(layer_above.eps_l)
    # @show layer.pad

    if layer.padrule == :none   # TODO even if pad were none we are never using this!
        # TODO does this work? and test if previous layer is img formatted (one of input, conv, maxpooling)
        # fill!(layer.pad_above_eps, ELT(0.0))
        thish, thisw, _, _ = size(layer_above.eps_l)
        belowh, beloww, _, _ = size(layer.pad_above_eps)  # bigger
        padh = div(belowh - thish, 2)
        padw = div(beloww - thisw, 2)
        # @show padh, padw
        @views pad_above_eps[begin+padh:end-padh, begin+padw:end-padw, :, :] .= eps_l_above
    elseif layer.padrule == :same
        pad_above_eps .= eps_l_above
    end

    layer.normalization_gradf(layer, layer_above, current_batch_size) # either noop or batchnorm_grad! TODO: should this receive the layer_above as input????

    @turbo for b in cb_rng    # axes(layer.eps_l, 4)     #   axes(layer.eps_l, 4)  mb_rng
        for oc in axes(layer.weight, 4)
            for j = 1:W_out-(f_w-1)
                for i = 1:H_out-(f_h-1) # prevent filter from extending out of bounds
                    for ic in axes(layer.weight, 3)
                        for fj in axes(layer.weight, 2)
                            for fi in axes(layer.weight, 1)
                                # Flipped weight indices for backward pass: use weight[f_h-fi+1,f_w-fj+1] instead of weight[fi,fj]
                                layer.eps_l[i+fi-1, j+fj-1, ic, b] += layer.weight[f_h-fi+1, f_w-fj+1, ic, oc] * layer.pad_above_eps[i, j, oc, b]
                            end
                        end
                    end
                end
            end
        end
    end

    # compute gradients
    compute_grad_weight!(layer, inverse_n_samples, current_batch_size)
    cb = current_batch_size

    if layer.dobias
        # replace with explicit loop for optimization   
        # layer.grad_bias .= reshape(sum(eps_l_above, dims=(1, 2, 4)), oc) .* inverse_n_samples

        fill!(layer.grad_bias, ELT(0.0))
        @turbo for b in cb_rng
            for j in axes(eps_l_above, 2)
                for i in axes(eps_l_above, 1)
                    for c in axes(eps_l_above, 3)
                        layer.grad_bias[c] += eps_l_above[i, j, c, b]
                    end
                end
            end
        end
        layer.grad_bias .*= inverse_n_samples
    end

    return     # nothing
end


function compute_grad_weight!(layer, inverse_n_samples, current_batch_size)
    H_out, W_out, _, mb = size(layer.eps_l)
    cb = current_batch_size
    cb_rng = 1:cb
    if mb == cb
        eps_l = layer.eps_l
        grad_weight = layer.grad_weight
        a_below = layer.a_below
        pad_a_below = layer.pad_a_below
    else
        eps_l = view(layer.eps_l, :, :, :, cb_rng)
        grad_weight = view(layer.grad_weight, :, :, :, cb_rng)
        a_below = view(layer.a_below, :, :, :, cb_rng)
        pad_a_below = view(layer.pad_a_below, :, :, :, cb_rng)
    end
    # @show size(layer.pad_a_below)
    # @show size(layer.a_below)

    # Initialize grad_weight to zero
    # fill!(layer.grad_weight, ELT(0.0)) # no allocations; faster than assignment
    if layer.padrule == :same  # remarkably, this works
        pad = layer.pad
        @views pad_a_below[begin+pad:end-pad, begin+pad:end-pad, :, :] .= a_below
        use_a_below = a_below   # no allocation
    else  # when padding is :none (e.g., 0) then ...
        use_a_below = a_below   # set alias to a_below to use in loop below.  no allocation
    end

    # Use @views to avoid copying subarrays
    @inbounds for oc in axes(eps_l, 3)      # 1:out_channels
        # View of the error for this output channel (all spatial positions, all batches)
        err = @view eps_l[:, :, oc, :]      # size H_out × W_out × batch_size

        # @show size(err)
        # @show size(layer.weight)
        # @show size(use_a_below)

        for ic in axes(use_a_below, 3)          # layer.pad_a_below, 
            # View of the input activation for this channel
            # (We'll slide this view for each filter offset)

            input_chan = @view use_a_below[:, :, ic, :]   # size H_in × W_in × batch_size
            # @show size(input_chan)
            for fj in axes(layer.weight, 2)
                for fi in axes(layer.weight, 1)
                    # Extract the overlapping region of input corresponding to eps_l[:, :, oc, :]
                    local_patch = @view input_chan[fi:fi+H_out-1, fj:fj+W_out-1, :]
                    # Accumulate gradient for weight at (fi,fj, ic, oc)
                    # layer.grad_weight[fi, fj, ic, oc] += sum(l * e for (l, e) in zip(local_patch, err))

                    # attempt proposed optimization
                    accum = ELT(0.0)
                    @turbo for idx in eachindex(local_patch, err)
                        accum += local_patch[idx] * err[idx]
                    end
                    layer.grad_weight[fi, fj, ic, oc] += accum

                end
            end
        end
    end

    # Average over batch (divide by batch_size)
    layer.grad_weight .*= inverse_n_samples
    return   # nothing
end


# MaxPoolLayer

# note: this implicitly assumes stride is the size of the patch
# functor style call for feedforward
"""
    (layer::MaxPoolLayer)(x::Array{ELT, 4}, current_batch_size::Int)

Perform the forward pass of a max pooling layer on a 4D input tensor.

# Arguments
- `layer::MaxPoolLayer`: The max pooling layer containing pooling parameters, so the argument is both function call and input argument.
- `x::Array{ELT, 4}`: Input tensor with dimensions (height, width, channels, batch_size)
- `current_batch_size::Int`: Size of batch, which must be equal to or smaller than the batch_sized used to define layers


# Returns
Nothing. Results are stored in-place in `layer.a`. The positions of maximum values
are recorded in `layer.mask` for use during backpropagation.

# Note
This implementation assumes the stride equals the pool size, resulting in non-overlapping
pooling windows.
"""
function (layer::MaxPoolLayer)(x::Array{ELT,4}, current_batch_size::Int)
    (pool_h, pool_w) = layer.pool_size
    (H_out, W_out, C, _) = size(layer.a)
    cb = current_batch_size
    # mb_rng = layer.mb_rng[]
    # re-initialize
    fill!(layer.a, ELT(0.0))
    fill!(layer.mask, false)

    # no stride: the pool window moves across the image edge to edge with no overlapping
    for bn = 1:cb      #  mb_rng
        for c = 1:C
            for j = 1:W_out
                for i = 1:H_out
                    # Find max value directly without creating a view
                    max_val = typemin(ELT)
                    max_a = 1
                    max_b = 1
                    for b = 1:pool_w
                        for a = 1:pool_h
                            val = x[pool_h*(i-1)+a, pool_w*(j-1)+b, c, bn]
                            if val > max_val
                                max_val = val
                                max_a = a
                                max_b = b
                            end
                        end
                    end
                    layer.a[i, j, c, bn] = max_val
                    # Set mask for max position only
                    layer.mask[pool_h*(i-1)+max_a, pool_w*(j-1)+max_b, c, bn] = true
                end
            end
        end
    end
    return
end

# functor-style call for back propagation
"""
    (layer::MaxPoolLayer)(layer_above, current_batch_size::Int)

Perform the backward pass (backpropagation) for a max pooling layer.

# Arguments
- `layer::MaxPoolLayer`: The max pooling layer containing pooling parameters and mask, so the argument is both function call and input argument.
- `layer_above`: The layer above in the network, containing error gradients
- `current_batch_size::Int`: Size of batch, which must be equal to or smaller than the batch_sized used to define layers


# Returns
Nothing. Error gradients are propagated through the max pooling layer and stored
in-place in `layer.eps_l`, with gradients routed only through the positions that
were selected during the forward pass (as recorded in `layer.mask`).

# Note
This implementation ensures that gradients flow only through the maximum value
positions identified during the forward pass.
"""
function (layer::MaxPoolLayer)(layer_above, current_batch_size::Int)
    cb = current_batch_size
    fill!(layer.eps_l, ELT(0.0))
    (pool_h, pool_w) = layer.pool_size
    # mb_rng = layer.mb_rng[]
    @inbounds for bn in 1:cb  # axes(layer_above.eps_l, 4)    #   mb_rng   # @inbounds
        for c in axes(layer.eps_l, 3)
            for j = axes(layer_above.eps_l, 2)  #  1:W_in
                for i = axes(layer_above.eps_l, 1) #  1:H_in
                    for b = 1:pool_w, a = 1:pool_h   # gives 2 rows and 2 columns for each i, j
                        if layer.mask[pool_h*(i-1)+a, pool_w*(j-1)+b, c, bn]
                            layer.eps_l[pool_h*(i-1)+a, pool_w*(j-1)+b, c, bn] = layer_above.eps_l[i, j, c, bn]
                            break
                        end
                    end
                end
            end
        end
    end
    return    # nothing
end


# FlattenLayer
# functor-style call for feedforward
"""
    (layer::FlattenLayer)(x::AbstractArray{ELT, 4}, current_batch_size::Int)

Perform the forward pass of a flatten layer, converting a 4D tensor to a 2D matrix.

# Arguments
- `layer::FlattenLayer`: The flatten layer, so the argument is both function call and input argument.
- `x::AbstractArray{ELT, 4}`: Input tensor with dimensions (height, width, channels, batch_size)
- `current_batch_size::Int`: Size of batch, which must be equal to or smaller than the batch_sized used to define layers


# Returns
Nothing. Results are stored in-place in `layer.a` with dimensions (height*width*channels, batch_size).

# Note
This implementation preserves batch dimension while flattening the spatial and channel dimensions.
"""
function (layer::FlattenLayer)(x::AbstractArray{ELT,4}, current_batch_size::Int)
    cb = current_batch_size
    h, w, ch, _ = size(x)
    # mb_rng = layer.mb_rng[]
    for b in 1:cb  #  axes(x, 4)    #  mb_rng  # iterate over batch dimension
        @turbo for c in axes(x, 3)  # iterate over channels
            c_offset = (c - 1) * h * w
            for j in axes(x, 2)  # iterate over width
                j_offset = (j - 1) * h
                for i in axes(x, 1)  # iterate over height
                    idx = c_offset + j_offset + i
                    # println("idx ", idx, " tdx ", tdx)
                    layer.a[idx, b] = x[i, j, c, b]
                end
            end
        end
    end
    return
end


"""
    (layer::FlattenLayer)(layer_above::LinearLayer, current_batch_size::Int)

Perform the backward pass (backpropagation) for a flatten layer.

# Arguments
- `layer::FlattenLayer`: The flatten layer, so the argument is both function call and input argument.
- `layer_above::LinearLayer`: The linear layer above, containing error gradients
- `current_batch_size::Int`: Size of batch, which must be equal to or smaller than the batch_sized used to define layers


# Returns
Nothing. Error gradients are reshaped from the 2D format of the linear layer back
to the 4D format of convolutional layers and stored in-place in `layer.eps_l`.

# Note
This implementation uses matrix multiplication with the transposed weights of the
layer above to compute gradients, then reshapes them to the original 4D format.
"""
function (layer::FlattenLayer)(layer_above::LinearLayer, current_batch_size::Int)
    cb = current_batch_size
    cb_rng = 1:cb

    # mb_rng = layer.mb_rng[]
    # Use pre-allocated dl_dflat array for matrix multiplication

    # @show size(layer.dl_dflat[:,mb_rng])
    # @show size(layer_above.weight')
    # @show size(layer_above.eps_l)

    @views mul!(layer.dl_dflat[:, cb_rng], layer_above.weight', layer_above.eps_l[:, cb_rng])  # in-place matrix multiplication

    # Reshape in-place using explicit indexing
    h, w, ch, _ = size(layer.eps_l)
    for b in cb_rng    # 1:batch_size   #   mb_rng
        idx = 1
        @turbo for c in 1:ch
            for j in 1:w
                for i in 1:h
                    layer.eps_l[i, j, c, b] = layer.dl_dflat[idx, b]
                    idx += 1
                end
            end
        end
    end
    return
end


# LinearLayer functors

"""
    (layer::LinearLayer)(x::Matrix{ELT}, current_batch_size::Int)

Perform the forward pass of a linear (fully connected) layer on a 2D input matrix.

# Arguments
- `layer::LinearLayer`: The linear layer containing weights, biases, and other parameters, so the argument is both function call and input argument.
- `x::Matrix{ELT}`: Input matrix with dimensions (features, batch_size)
- `current_batch_size::Int`: Size of batch, which must be equal to or smaller than the batch_sized used to define layers


# Returns
Nothing. Results are stored in-place in `layer.a` after applying weights, biases,
normalization, and activation functions.

# Note
This implementation uses in-place matrix multiplication and optimized broadcasting
for bias addition to minimize memory allocations.
"""
function (layer::LinearLayer)(x::Matrix{ELT}, current_batch_size::Int)
    cb = current_batch_size
    cb_rng = 1:cb
    mb = size(layer.z, 2)
    if cb == mb  # no views: pre-allocated layer arrays are sized for standard minibatch
        z = layer.z  # no cost to setting an alias to existing memory
        x = x
        a_below = layer.a_below
    else   # use views for the last minibatch if it is fraction of standard
        z = view(layer.z, :, cb_rng)  # TODO we really only need to do this once for a given layer
        x = view(x, :, cb_rng)
        a_below = view(layer.a_below, :, cb_rng)
    end
    # @views mul!(layer.z[:, cb_rng], layer.weight, x[:, cb_rng])  # in-place matrix multiplication
    mul!(z, layer.weight, x)

    # bias: explicit loop faster than broadcasting
    if layer.dobias
        @turbo for j in cb_rng   # axes(layer.z, 2)   #   mb_rng # For each column (sample)
            for i in axes(layer.z, 1)  # For each row (output neuron)
                # layer.z[i, j] += layer.bias[i]
                z[i, j] += layer.bias[i]
            end
        end
    end

    # @views layer.a_below[:, cb_rng] .= x[:, cb_rng]   # assign alias for using in backprop
    @turbo a_below .= x

    layer.normalizationf(layer, current_batch_size) # either batchnorm! or noop

    layer.activationf(layer, current_batch_size)  # updates layer.a; see relu!, leaky_relu!, etc.
    return
end


"""
    (layer::LinearLayer)(layer_above::LinearLayer, currrent_batch_size::Int)

Perform the backward pass (backpropagation) for a linear (fully connected) layer.

# Arguments
- `layer::LinearLayer`: The linear layer containing weights, biases, and other parameters, so the argument is both function call and input argument.
- `layer_above::LinearLayer`: The layer above in the network, containing error gradients
- `current_batch_size::Int`: Size of batch, which must be equal to or smaller than the batch_sized used to define layers


# Returns
Nothing. Gradients are computed and stored in-place in `layer.grad_weight`,
`layer.grad_bias` (if `layer.dobias` is true), and `layer.eps_l` (for propagating
errors to lower layers).

# Note
This implementation handles both output and hidden layers differently, applying
appropriate activation gradients and normalization gradients as needed.
"""
function (layer::LinearLayer)(layer_above::Union{LinearLayer,InputLayer}, current_batch_size::Int)  # layer_above::LinearLayer
    # for the outputlayer, layer_above is actually layer below
    # TODO it might be nice to have a separate method for the output layer: how to dispatch on it?
    cb = current_batch_size
    cb_rng = 1:cb
    mb = size(layer.eps_l, 2)
    if cb == mb  # just deref arrays in layer struct 
        eps_l = layer.eps_l
        a_below = layer.a_below
        layer.isoutput || (eps_l_above = layer_above.eps_l) # ugly but very little performance penalty
        grad_a = layer.grad_a
    else  # use views for the last minibatch 
        eps_l = view(layer.eps_l, :, cb_rng)
        a_below = view(layer.a_below, :, cb_rng)
        layer.isoutput || (eps_l_above = view(layer_above.eps_l, :, cb_rng))
        grad_a = view(layer.grad_a, :, cb_rng)
    end
    inverse_n_samples = ELT(1.0) / ELT(cb)
    if layer.isoutput
        mul!(layer.grad_weight, eps_l, a_below')
        @turbo layer.grad_weight .*= inverse_n_samples  # in-place scaling
    else  # this is hidden layer
        layer.activation_gradf(layer, current_batch_size)  # calculates layer.grad_a
        mul!(eps_l, layer_above.weight', eps_l_above)
        @turbo eps_l .*= grad_a

        layer.normalization_gradf(layer, current_batch_size) # either noop or batchnorm_grad!

        mul!(layer.grad_weight, eps_l, a_below')
        layer.grad_weight .*= inverse_n_samples # in-place scaling
    end

    # Compute bias gradient efficiently without allocations
    if layer.dobias
        fill!(layer.grad_bias, ELT(0.0))
        @turbo for j in cb_rng   # axes(layer.eps_l, 2)  # axes(layer.eps_l, 2)  mb_rng
            for i in axes(layer.eps_l, 1)
                layer.grad_bias[i] += layer.eps_l[i, j]
            end
        end
        layer.grad_bias .*= inverse_n_samples  # in-place scaling
    end
    return
end
