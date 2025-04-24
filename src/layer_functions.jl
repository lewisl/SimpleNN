"""
    struct LayerSpec

Provide input parameters to define a layer of a neural network model.
These parameters are the union of all parameters needed across many
types of layers. There are convenience methods that limit the inputs
to only the parameters needed for a specific type of layer.

These inputs are used to fully define each type of layer including
the weights and arrays required during model training.
"""
Base.@kwdef struct LayerSpec
    name::Symbol = :noname
    kind::Symbol = :none
    activation::Symbol = :none
    adj::Float64 = 0      # leaky_relu factor. also for he_initialize
    h::Int64 = 0          # image height (rows) or output neurons for linear layers
    w::Int64 = 0          # image width (columns)
    outch::Int64 = 0
    f_h::Int64 = 0        # filter height (rows)
    f_w::Int64 = 0        # filter width (columns)
    inch::Int64 = 0
    padrule::Symbol = :same       # either :same or :none
    stride::Int64 = 1             # no input required to accept default
end

# LayerSpec methods for specific kinds of layers
"""
    convlayerspec(;name::Symbol, activation::Symbol, adj::Float64=0.002, h::Int64=0, w::Int64=0, outch::Int64=0, f_h::Int64, f_w::Int64, inch::Int64=0, padrule::Symbol=:same)

Only inputs needed for a convlayer are passed to the LayerSpec. 
Note that h, w, and inch will be calculated from the previous layer,
which should be an image input, another conv layer, or a maxpooling layer.
You must provide inputs for name, activation, outch, f_h, and f_w.
"""
function convlayerspec(; name::Symbol, activation::Symbol=:relu, adj::Float64=0.002, h::Int64=0, w::Int64=0, outch::Int64, f_h::Int64, f_w::Int64, inch::Int64=0, padrule::Symbol=:same)
    LayerSpec(name=name, kind=:conv, activation=activation, adj=adj, h=h, w=w, outch=outch, f_h=f_h, f_w=f_w, inch=inch, padrule=padrule)
end

function linearlayerspec(; name::Symbol, activation::Symbol=:relu, adj::Float64=0.002, output::Int64)
    LayerSpec(name=name, kind=:linear, activation=activation, adj=adj, h=output)
end

function maxpoollayerspec(; name::Symbol, f_h::Int, f_w::Int)
    LayerSpec(name=name, kind=:maxpool, f_h=f_h, f_w=f_w)
end

function flattenlayerspec(; name::Symbol)
    LayerSpec(name=name, kind=:flatten)
end


# ============================
# Mutable structs for layers: hold weights, bias, data storage, dims
# ============================


Base.@kwdef mutable struct ConvLayer
    name::Symbol = :noname
    activationf::Function = relu!
    activation_gradf::Function = relu_grad!
    adj::Float64 = 0.0
    weight::Array{Float64,4} = Float64[;;;;]  # (filter_h, filter_w, in_channels, out_channels)
    padrule::Symbol = :same   # other option is :none
    stride::Int64 = 1     # assume stride is symmetrical for now
    bias::Vector{Float64} = Float64[]    # (out_channels)
    dobias::Bool = true
    z::Array{Float64,4} = Float64[;;;;]
    pad_x::Array{Float64,4} = Float64[;;;;]
    a::Array{Float64,4} = Float64[;;;;]
    a_below::Array{Float64,4} = Float64[;;;;]
    pad_a_below::Array{Float64,4} = Float64[;;;;]
    eps_l::Array{Float64,4} = Float64[;;;;]
    pad_next_eps::Array{Float64,4} = Float64[;;;;]  # TODO need to test if this is needed given successive conv layer sizes
    grad_a::Array{Float64,4} = Float64[;;;;]
    grad_weight::Array{Float64,4} = Float64[;;;;]
    grad_bias::Vector{Float64} = Float64[]
end

# method to do prep calculations based on LayerSpec inputs, then create a ConvLayer
function ConvLayer(lr::LayerSpec, prevlayer, n_samples)
    outch = lr.outch
    prev_h, prev_w, inch, _ = size(prevlayer.a)

    pad = ifelse(lr.padrule == :same, 1, 0)
    # output image dims: calculated once rather than over and over in training loop
    out_h = div((prev_h + 2pad - lr.f_h), lr.stride) + 1
    out_w = div((prev_w + 2pad - lr.f_w), lr.stride) + 1
    ConvLayer(
        name=lr.name,
        activationf = if lr.activation == :relu
                relu!
            elseif lr.activation == :leaky_relu
                leaky_relu!
            elseif lr.activation == :none
                noop
            else
                error("Only :relu, :leaky_relu and :none  supported, not $(Symbol(lr.activation)).")
            end,
        activation_gradf = if lr.activation == :relu
                relu_grad!
            elseif lr.activation == :leaky_relu
                leaky_relu_grad!
            elseif lr.activation == :none
                noop
            else
                error("Only :relu, :leaky_relu and :none  supported, not $(Symbol(lr.activation)).")
            end,
        adj=lr.adj,
        weight=he_initialize((lr.f_h, lr.f_w, inch, lr.outch), scale=2.2, adj=lr.adj),
        padrule=lr.padrule,
        stride=lr.stride,
        bias=zeros(outch),
        pad_x=zeros(out_h + 2pad, out_w + 2pad, inch, n_samples),
        pad_a_below=zeros(prev_h + 2pad, prev_w + 2pad, inch, n_samples),
        z=zeros(out_h, out_w, outch, n_samples),
        a=zeros(out_h, out_w, outch, n_samples),
        eps_l=zeros(prev_h, prev_w, inch, n_samples),
        grad_a=zeros(out_h, out_w, outch, n_samples),
        pad_next_eps=zeros(prev_h, prev_w, outch, n_samples),
        grad_weight=zeros(lr.f_h, lr.f_w, inch, outch),
        grad_bias=zeros(outch),
    )
end


Base.@kwdef mutable struct LinearLayer
    name::Symbol = :noname
    activationf::Function = relu!
    activation_gradf::Function = relu_grad!
    adj::Float64 = 0.0
    weight::Array{Float64,2} = Float64[;;] # (output_dim, input_dim)
    output_dim::Int64 = 0
    input_dim::Int64 = 0
    bias::Vector{Float64} = Float64[]     # (output_dim)
    dobias::Bool = true
    z::Array{Float64,2} = Float64[;;]       # feed forward linear combination result 
    a::Array{Float64,2} = Float64[;;]      # feed forward activation output
    # ALIAS TO ACTIVATION FROM LOWER BELOW used in backprop, but don't make a copy
    # type and size don't matter for an alias
    # we don't need to pre-allocate this. It just simplifies the API
    # when do we assign this?????  in layer_forward for LinearLayer
    a_below::Array{Float64,2} = Float64[;;]
    eps_l::Array{Float64,2} = Float64[;;]   # backprop error of the layer
    grad_a::Array{Float64,2} = Float64[;;]  # backprop derivative of activation output
    grad_weight::Array{Float64,2} = Float64[;;]
    grad_bias::Vector{Float64} = Float64[]
end

# method to do prep calculations based on LayerSpec inputs, then create a LinearLayer
function LinearLayer(lr::LayerSpec, prevlayer, n_samples)
    # weight dims
    outputs = lr.h        # rows
    inputs = prevlayer.output_dim    # columns
    LinearLayer(
        name=lr.name,
        activationf = if lr.activation == :relu
                relu!
            elseif lr.activation == :leaky_relu
                leaky_relu!
            elseif lr.activation == :none
                noop
            elseif lr.activation == :softmax
                softmax!
            elseif lr.activation == :logistic   # rarely used any more
                logistic!
            elseif lr.activation == :regression
                regression!
            else
                error("Only :relu, :leaky_relu, :softmax and :none  supported, not $(Symbol(lr.activation)).")
            end,
        activation_gradf = if lr.activation == :relu  # this has no effect on the output layer, but need it for hidden layers
                relu_grad!
            elseif lr.activation == :leaky_relu
                leaky_relu_grad!
            elseif lr.activation == :softmax
                noop
            elseif lr.activation == :none
                noop
            else
                error("Only :relu, :leaky_relu, :softmax and :none  supported, not $(Symbol(lr.activation)).")
            end,
        adj=lr.adj,
        weight=he_initialize((outputs, inputs), scale=1.5, adj=lr.adj),
        output_dim=outputs,
        input_dim=inputs,
        bias=zeros(outputs),
        z=zeros(outputs, n_samples),
        a=zeros(outputs, n_samples),
        eps_l=zeros(outputs, n_samples),
        grad_a=zeros(outputs, n_samples),
        grad_weight=zeros(outputs, inputs),
        grad_bias=zeros(outputs))
end

# no weight, bias, gradients, activation
Base.@kwdef mutable struct FlattenLayer
    name::Symbol = :noname
    output_dim::Int64 = 0
    dl_dflat::Array{Float64,2} = Float64[;;]
    a::Array{Float64,2} = Float64[;;]
    eps_l::Array{Float64,4} = Float64[;;;;]
end

# method to prepare inputs and create layer
function FlattenLayer(lr::LayerSpec, prevlayer, n_samples)
    h, w, ch, _ = size(prevlayer.a)
    output_dim = h * w * ch

    FlattenLayer(
        name=lr.name,
        output_dim=output_dim,
        a=zeros(output_dim, n_samples),
        dl_dflat=zeros(output_dim, n_samples),
        eps_l=zeros(h, w, ch, n_samples)
    )
end

Base.@kwdef mutable struct InputLayer     # we only have this to simplify feedforward loop
    name::Symbol = :noname
    kind::Symbol = :image   # other allowed value is :linear
    out_h::Int64 = 0
    out_w::Int64 = 0
    outch::Int64 = 0
    a::Array{Float64}   # no default provided because dims different for :image vs :linear
end


Base.@kwdef mutable struct MaxPoolLayer
    name::Symbol = :noname
    pool_size::Tuple{Int,Int}
    a::Array{Float64,4} = Float64[;;;;]
    mask::Array{Bool,4} = Bool[;;;;]
    eps_l::Array{Float64,4} = Float64[;;;;]
end

# method to prepare inputs and create layer
function MaxPoolLayer(lr::LayerSpec, prevlayer, n_samples)

    in_h, in_w, outch, _ = size(prevlayer.grad_a)
    out_h = div(in_h, lr.f_h) # assume stride = lr.f_h implicit in code
    out_w = div(in_w, lr.f_w)  # ditto
    batch_size = n_samples

    MaxPoolLayer(
        name=lr.name,
        pool_size=(lr.f_h, lr.f_w),
        a=zeros(out_h, out_w, outch, batch_size),
        mask=falses(in_h, in_w, outch, batch_size),
        eps_l=zeros(in_h, in_w, outch, batch_size),
    )
end

Base.@kwdef mutable struct stat_series
    acc::Array{Float64,1} = Float64[]
    cost::Array{Float64,1} = Float[]
    batch_size::Int = 0
    epochs::Int = 0
    minibatch_size = 0
end


# =====================
# ConvLayer
# =====================

# feedforward layers are all called layer_forward! and dispatch on the layer type
# backprop layers are all called layer_backward! and dispatch on the layer type

function layer_forward!(layer::ConvLayer, x::AbstractArray{Float64,4}, batch_size)
    layer.a_below = x   # as an alias, this might not work for backprop though it seems to

    fill!(layer.z, 0.0)

    if layer.padrule == :same  # we know padding = 1 for :same
        @views layer.pad_x[begin+1:end-1, begin+1:end-1, :, :] .= x
    else
        layer.pad_x .= x
    end

    @inbounds for b in axes(layer.z, 4)
        for oc in axes(layer.z, 3)  # this could also be axes(layer.weight, 4)
            for j in axes(layer.z, 2)
                for i in axes(layer.z, 1)
                    for ic in axes(layer.weight, 3)
                        for fw in axes(layer.weight, 2)
                            for fh in axes(layer.weight, 1)
                                layer.z[i, j, oc, b] += (layer.pad_x[i+fh-1, j+fw-1, ic, b]
                                                         *
                                                        layer.weight[fh, fw, ic, oc])
                            end  # columns of filter
                        end  # rows of filter
                    end  # input channels
                    layer.z[i, j, oc, b] += layer.bias[oc]
                end  # output column pixels
            end   # output row pixels
        end  # output channels
    end   # each sample in the minibatch or training set

    # activation
    layer.activationf(layer)

    return
end


function layer_backward!(layer::ConvLayer, layer_next, n_samples)
    f_h, f_w, _, _ = size(layer.weight)
    H_out, W_out, _, _ = size(layer.eps_l)

    fill!(layer.grad_weight, 0.0)  # reinitialization to allow accumulation of convolutions
    fill!(layer.eps_l, 0.0)
    fill!(layer.grad_a, 0.0)

    layer.activation_gradf(layer)

    # @show size(layer_next.eps_l)
    # @show size(layer.grad_a)

    layer_next.eps_l .*= layer.grad_a

    if layer.padrule == :none   # TODO even if pad were none we are never using this!
        # TODO does this work? and test if previous layer is img formatted (one of input, conv, maxpooling)
        fill!(layer.pad_next_eps, 0.0)
        @views layer.pad_next_eps[2:end-1, 2:end-1, :, :] .= layer_next.eps_l
    elseif layer.padrule == :same
        # nothing to do: arrays are going to come out the right size
    end

    @inbounds for b in axes(layer.eps_l, 4)
        for oc in axes(layer.weight, 4)
            for i = 1:H_out-(f_h-1) # prevent filter from extending out of bounds
                for j = 1:W_out-(f_w-1)
                    for ic in axes(layer.weight, 3)
                        for fj in axes(layer.weight, 2)
                            for fi in axes(layer.weight, 1)
                                # Flipped weight indices for backward pass: use weight[f_h-fi+1,f_w-fj+1] instead of weight[fi,fj]
                                layer.eps_l[i+fi-1, j+fj-1, ic, b] += layer.weight[f_h-fi+1, f_w-fj+1, ic, oc] * layer.pad_next_eps[i, j, oc, b]
                            end
                        end
                    end
                end
            end
        end
    end


    # compute gradients
    compute_grad_weight!(layer, n_samples)

    # Compute bias gradients
    for oc = axes(layer.eps_l, 3) # the channels axis
        @views layer.grad_bias[oc] = sum(layer.eps_l[:, :, oc, :]) .* (1.0 / Float64(n_samples))
    end

    return     # nothing
end

function compute_grad_weight!(layer, n_samples)
    H_out, W_out, _, _ = size(layer.eps_l)

    # Initialize grad_weight to zero
    fill!(layer.grad_weight, 0.0) # no allocations; faster than assignment
    if layer.padrule == :same
        fill!(layer.pad_a_below, 0.0)
        @views layer.pad_a_below[2:end-1, 2:end-1, :, :] .= layer.a_below
    else
        layer.pad_a_below = layer.a_below  # set alias to a_below to use in loop below.  no allocation
    end

    # Use @views to avoid copying subarrays
    @inbounds for oc in axes(layer.eps_l, 3)      # 1:out_channels
        # View of the error for this output channel (all spatial positions, all batches)
        err = @view layer.eps_l[:, :, oc, :]      # size H_out × W_out × batch_size
        for ic in axes(layer.pad_a_below, 3)          # 1:in_channels
            # View of the input activation for this channel
            # (We'll slide this view for each filter offset)
            input_chan = @view layer.pad_a_below[:, :, ic, :]   # size H_in × W_in × batch_size
            for fj in axes(layer.weight, 2)
                for fi in axes(layer.weight, 1)
                    # Extract the overlapping region of input corresponding to eps_l[:, :, oc, :]
                    local_patch = @view input_chan[fi:fi+H_out-1, fj:fj+W_out-1, :]
                    # Accumulate gradient for weight at (fi,fj, ic, oc)
                    layer.grad_weight[fi, fj, ic, oc] += sum(l * e for (l, e) in zip(local_patch, err))
                end
            end
        end
    end

    # Average over batch (divide by batch_size)
    layer.grad_weight .*= (1 / n_samples)
    return   # nothing
end

# =====================
# MaxPoolLayer
# =====================


# note: this implicitly assumes stride is the size of the patch
# tested: it works
function layer_forward!(layer::MaxPoolLayer, x::Array{Float64,4}, n_samples)
    # layer.input_shape = size(x)
    (pool_h, pool_w) = layer.pool_size
    (H_out, W_out, C, B) = size(layer.a)
    # re-initialize
    fill!(layer.a, 0.0)
    fill!(layer.mask, false)

    # no stride: the pool window moves across the image edge to edge with no overlapping
    @inbounds for bn = 1:B
        for c = 1:C
            for j = 1:W_out
                for i = 1:H_out
                    region = view(x, (pool_h*(i-1)+1):(pool_h*i), (pool_w*(j-1)+1):(pool_w*j), c, bn)
                    max_val = maximum(region)
                    layer.a[i, j, c, bn] = max_val
                    for a = 1:pool_h, b = 1:pool_w
                        if region[a, b] == max_val && !layer.mask[pool_h*(i-1)+a, pool_w*(j-1)+b, c, bn]
                            layer.mask[pool_h*(i-1)+a, pool_w*(j-1)+b, c, bn] = true
                            break
                        end
                    end
                end
            end
        end
    end
    return
end


function layer_backward!(layer::MaxPoolLayer, layer_next, n_samples)
    fill!(layer.eps_l, 0.0)
    (pool_h, pool_w) = layer.pool_size
    @inbounds for bn in axes(layer_next.eps_l, 4)
        for c in axes(layer.eps_l, 3)
            for j = axes(layer_next.eps_l, 2)  #  1:W_in
                for i = axes(layer_next.eps_l, 1) #  1:H_in
                    for b = 1:pool_w, a = 1:pool_h   # gives 2 rows and 2 columns for each i, j
                        if layer.mask[pool_h*(i-1)+a, pool_w*(j-1)+b, c, bn]
                            layer.eps_l[pool_h*(i-1)+a, pool_w*(j-1)+b, c, bn] = layer_next.eps_l[i, j, c, bn]
                            break
                        end
                    end
                end
            end
        end
    end
    return    # nothing
end


# =====================
# FlattenLayer
# =====================

function layer_forward!(layer::FlattenLayer, x::Array{Float64,4}, batch_size)
    @inbounds for idx in axes(x, 4)  # iterate over batch dimension (4th dimension)
        @views layer.a[:, idx] .= x[:, :, :, idx][:]  # Flatten the first 3 dimensions and assign to `layer.a`
    end
    return
end


function layer_backward!(layer::FlattenLayer, layer_next::LinearLayer, batch_size)
    # Use pre-allocated dl_dflat array for matrix multiplication
    mul!(layer.dl_dflat, layer_next.weight', layer_next.eps_l)  # in-place matrix multiplication
    # Reshape in-place using views
    h, w, ch, _ = size(layer.eps_l)
    @inbounds for i in 1:batch_size
        @views layer.eps_l[:, :, :, i] .= reshape(layer.dl_dflat[:, i], h, w, ch)
    end
    return
end


# =====================
# LinearLayer
# =====================

function layer_forward!(layer::LinearLayer, x::Matrix{Float64}, batch_size)
    mul!(layer.z, layer.weight, x)  # in-place matrix multiplication
    # layer.z .+= layer.bias  # in-place addition with broadcasting

    # Replace broadcasting with explicit loop
    for j in axes(layer.z, 2)  # For each column (sample)
        for i in axes(layer.z, 1)  # For each row (output neuron)
            layer.z[i, j] += layer.bias[i]
        end
    end

    layer.a_below = x   # assign alias for using in backprop
    layer.activationf(layer)
    return
end

function layer_backward!(layer::LinearLayer, layer_next::LinearLayer, n_samples; output=false)
    if output
        # layer.eps_l calculated by prior call to dloss_dz
        mul!(layer.grad_weight, layer.eps_l, layer.a_below')  # in-place matrix multiplication
        layer.grad_weight .*= (1.0 / n_samples)  # in-place scaling
    else  # this is hidden layer
        layer.activation_gradf(layer)  # calculates layer.grad_a
        mul!(layer.eps_l, layer_next.weight', layer_next.eps_l)  # in-place matrix multiplication
        layer.eps_l .*= layer.grad_a  # in-place element-wise multiplication
        mul!(layer.grad_weight, layer.eps_l, layer.a_below')  # in-place matrix multiplication
        layer.grad_weight .*= (1.0 / n_samples)  # in-place scaling
    end
    # Compute bias gradient efficiently without allocations
    fill!(layer.grad_bias, 0.0)  # Reset to zero
    for j in axes(layer.eps_l, 2)  # Iterate over columns (batch dimension)
        for i in axes(layer.eps_l, 1)  # Iterate over rows (output dimension)
            layer.grad_bias[i] += layer.eps_l[i, j]
        end
    end
    layer.grad_bias .*= (1.0 / n_samples)  # in-place scaling
    return     # nothing
end


# =====================
# activation functions
# =====================

function relu!(layer)
    @inbounds @fastmath begin
        for i in eachindex(layer.z)
            # Directly compare and assign, avoiding any temporary allocations
            layer.a[i] = ifelse(layer.z[i] >= 0.0, layer.z[i], 0.0)
        end
    end
end

function leaky_relu!(layer)
    @inbounds @fastmath begin
        for i in eachindex(layer.z)
            # Directly compare and assign, avoiding any temporary allocations
            layer.a[i] = ifelse(layer.z[i] >= 0.0, layer.z[i], layer.adj * layer.x[i])
        end
    end
end


# use for activation of conv or linear, when activation is requested as :none
function noop(args...)
end

function relu_grad!(layer)   # I suppose this is really leaky_relu...
    @inbounds for i = eachindex(layer.z)  # when passed any array, this will update in place
        layer.grad_a[i] = ifelse(layer.z[i] > 0.0, 1.0, 0.0)  # prevent vanishing gradients by not using 0.0
    end
end

function leaky_relu_grad!(layer)   # I suppose this is really leaky_relu...
    @inbounds for i = eachindex(layer.z)  # when passed any array, this will update in place
        layer.grad_a[i] = ifelse(layer.z[i] > 0.0, 1.0, layer.adj)  # prevent vanishing gradients by not using 0.0
    end
end


# =====================
# classifier and loss functions
# =====================

function dloss_dz!(layer, target)
    layer.eps_l .= layer.a .- target
end

# tested to have no allocations
function softmax!(layer, adj=0.0) # adj arg required for calling loop: not used
    for c in axes(layer.z, 2)
        va = view(layer.a, :, c)
        vz = view(layer.z, :, c)
        va .= exp.(vz .- maximum(vz))
        va .= va ./ (sum(va) .+ 1e-12)
    end
    return
end

function logistic!(layer, adj=0.0)
    @fastmath layer.a .= 1.0 ./ (1.0 .+ exp.(.-layer.z))  
end

function regression!(layer, adj=0.0)
    layer.a[:] = layer.z[:]
end
