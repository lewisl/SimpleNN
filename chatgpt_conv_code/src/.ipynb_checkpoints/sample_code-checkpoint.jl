#=
TODO
- calculate size of successive conv layers and linear layers
- add stride to loops and array size calculations

=#



using Random
using LinearAlgebra
using Colors, Plots
using Serialization
using MLDatasets
using StatsBase


# ============================
# Sample model definitions
# ============================

function conv_specs()
    input = LayerSpec(h=28, w=28, outch=1, kind=:input, name=:input)
    conv1 = LayerSpec(h=26, w=26, outch=4, f_h=3, f_w=3, inch=1, kind=:conv, name=:conv1, activation=:relu, adj=0.002)
    conv2 = LayerSpec(h=24, w=24, outch=8, f_h=3, f_w=3, inch=4, kind=:conv, name=:conv2, activation=:relu, adj=0.002)
    flatten = LayerSpec(h=24, w=24, outch=8, kind=:flatten, name=:flatten)
    linear1 = LayerSpec(h=2000, kind=:linear, activation=:relu, name=:linear1, adj=0.002)
    linear2 = LayerSpec(h=1000, kind=:linear, activation=:relu, name=:linear2, adj=0.002)
    linear3 = LayerSpec(h=500, kind=:linear, activation=:relu, name=:linear3, adj=0.002)
    linear4 = LayerSpec(h=100, kind=:linear, activation=:relu, name=:linear4, adj=0.002)
    linear5 = LayerSpec(h=10, kind=:linear,activation=:softmax, name=:output)

    layerspecs = LayerSpec[]
    push!(layerspecs, input, conv1, conv2, flatten, linear1, linear2, linear3, linear4, linear5)

    return layerspecs
end

function small_conv()
    input = LayerSpec(h=28, w=28, outch=1, kind=:input, name=:input)
    conv1 = LayerSpec(h=26, w=26, outch=16, f_h=3, f_w=3, inch=1, kind=:conv, name=:conv1, activation=:relu, adj=0.002)
    flatten = LayerSpec(h=26, w=26, outch=16, kind=:flatten, name=:flatten)
    linear1 = LayerSpec(h=256, kind=:linear, activation=:relu, name=:linear1, adj=0.002)
    output = LayerSpec(h=10, kind=:linear, activation=:softmax, name=:output)

    layerspecs = LayerSpec[]
    push!(layerspecs, input, conv1, flatten, linear1, output)

    return layerspecs
end

function simple_specs()
    input = LayerSpec(h=28, w=28, outch=1, kind=:input, name=:input)
    flatten = LayerSpec(h=28, w=28, outch=1, kind=:flatten, name=:flatten)
    linear1 = LayerSpec(h=200, kind=:linear, activation=:relu, name=:linear1, adj=0.01)
    linear2 = LayerSpec(h=200, kind=:linear, activation=:relu, name=:linear2, adj=0.01)
    linear3 = LayerSpec(h=10, kind=:linear, name=:output, activation=:softmax)

    layerspecs = LayerSpec[]
    push!(layerspecs, input,  flatten, linear1, linear2, linear3)

    return layerspecs
end

function deep_specs()
    input = LayerSpec(h=28, w=28, outch=1, kind=:input, name=:input)
    flatten = LayerSpec(h=28, w=28, outch=1, kind=:flatten, name=:flatten)
    linear1 = LayerSpec(h=600, kind=:linear, name=:linear1, activation=:relu, adj=0.0001)
    linear2 = LayerSpec(h=200, kind=:linear, name=:Linear2, activation=:relu, adj=0.0001)
    linear3 = LayerSpec(h=10, kind=:linear, name=:output, activation=:softmax)

    layerspecs = LayerSpec[]
    push!(layerspecs, input,  flatten, linear1, linear2, linear3)

    return layerspecs
end


# ============================
# Setup training
# ============================

@Base.kwdef struct LayerSpec
    name::Symbol=:noname
    kind::Symbol=:none
    activation::Symbol=:none
    adj::Float64=0  # leaky_relu factor. also for he_initialize
    h::Int64=0
    w::Int64=0
    outch::Int64=0
    f_h::Int64=0
    f_w::Int64=0
    inch::Int64=0
end


function preptrain(modelspecs::Function, batch_size, mini_batch_size)
    trainset = MNIST(:train)
    testset = MNIST(:test)

    # batch_size = 10000
    # minibatch_size = 50
    x_train = trainset.features[1:28, 1:28,1:batch_size]
    x_train = Float64.(x_train)
    x_train = reshape(x_train, 28, 28, 1, batch_size)
    @show size(x_train)

    y_train = trainset.targets[1:batch_size]
    y_train = indicatormat(y_train)
    y_train = Float64.(y_train)

    # shuffle the variables and outcome identically
    img_idx = shuffle(1:size(x_train,4))
    x_train_shuf = x_train[:,:,:,img_idx]
    y_train_shuf = y_train[:,img_idx]

    layerspecs = modelspecs()   # set_layer_specs()
    display(layerspecs);println()

    layers = allocate_layers(layerspecs, mini_batch_size);

    return layerspecs, layers, x_train_shuf, y_train_shuf
end


# ============================
# pre-alllocation of Layers
# ============================

function allocate_layers(lsvec::Vector{LayerSpec}, n_samples)

    Random.seed!(42)
    layerdat = []

    for (idx,lr)  in enumerate(lsvec)
        if idx == 1
            if lr.kind != :input
                error("First layer must be the input layer.")
            else
                push!(layerdat,
                    InputLayer(
                        name = lr.name,
                        kind = lr.kind,
                        out_h = lr.h,
                        out_w = lr.w,
                        out_c = lr.outch,
                        a = zeros(lr.h, lr.w, lr.outch, n_samples),
                        ))
            end
            continue  # skip the ifs and go to next lr
        elseif lr.kind == :conv
            # filter dims
            f_h = lr.f_h
            f_w = lr.f_w
            inch = lr.inch
            outch = lr.outch
            # output image dims
            out_h=lr.h
            out_w=lr.w
            # backprop dimensions
            b_out_h = lsvec[idx-1].h   
            b_out_w = lsvec[idx-1].w   
            b_out_c = lsvec[idx-1].outch   
            # @show lr.adj
            push!(layerdat,
                ConvLayer(  
                    name = lr.name,
                    activationf =   if lr.activation ==  :relu 
                                        relu!
                                    elseif lr.activation == :none
                                        noop
                                    else error("Only :relu, and :none  supported, not $(Symbol(lr.activation)).")
                                    end,
                    adj = lr.adj,
                    weight=he_initialize((f_h,f_w,inch,outch),scale=2.0, adj=lr.adj),
                    f_h=f_h,
                    f_w=f_w,
                    inch=inch,
                    outch=outch,
                    bias=zeros(outch),
                    z=zeros(out_h, out_w, outch, n_samples),
                    a=zeros(out_h, out_w, outch, n_samples),
                    out_h = out_h,
                    out_w = out_w,
                    in_h = b_out_h,
                    in_w = b_out_w,
                    eps_l=zeros(b_out_h, b_out_w, b_out_c, n_samples),  
                    grad_a=zeros(out_h, out_w, outch, n_samples),
                    pad_next_eps=zeros(b_out_h, b_out_w, outch, n_samples),
                    grad_weight=zeros(f_h, f_w, inch, outch),
                    grad_bias=zeros(outch),
                    b=n_samples  # not clear we need this!
                    ))   

        elseif lr.kind == :linear
            # weight dims
            outputs=lr.h
            inputs = layerdat[idx-1].output_dim
            # output dims
            # @show lr.adj
            push!(layerdat,
                LinearLayer(
                    name = lr.name,
                    activationf =   if lr.activation ==  :relu 
                                        relu!
                                    elseif lr.activation == :none
                                        noop
                                    elseif lr.activation == :softmax
                                        softmax!
                                    else error("Only :relu, :softmax and :none  supported, not $(Symbol(lr.activation)).")
                                    end,
                    adj = lr.adj,
                    weight = he_initialize((outputs, inputs), scale=1.0, adj=lr.adj),
                    output_dim =  outputs,
                    input_dim =   inputs,
                    bias = zeros(outputs),
                    z=zeros(outputs, n_samples),
                    a=zeros(outputs, n_samples),
                    eps_l = zeros(outputs, n_samples),
                    grad_a = zeros(outputs, n_samples),
                    grad_weight = zeros(outputs, inputs),
                    grad_bias = zeros(outputs),
                    ))

        elseif lr.kind == :maxpool
            # TODO

        elseif lr.kind == :flatten
            h = lsvec[idx-1].h
            w = lsvec[idx-1].w
            ch = lsvec[idx-1].outch
            output_dim = h*w*ch
            b = n_samples
            push!(layerdat,
                FlattenLayer(
                    name = lr.name,
                    h=h,
                    w=w,
                    ch=ch,
                    b=b;
                    output_dim=output_dim,
                    a=zeros(output_dim, n_samples),
                    dl_dflat = zeros(output_dim, n_samples),
                    eps_l=zeros(h,w,ch,n_samples)
                    ))
        else
            error("Found unrecognized layer kind")
        end
    end

    return layerdat
end


# ============================
# Setup prediction
# ============================

function setup_preds(modelspecs::Function, layers, n_samples)
    predlayerspecs = modelspecs()
    predlayers = allocate_layers(predlayerspecs, n_samples)
    for (prlr, lr) in zip(predlayers, layers)
        if (typeof(prlr) == ConvLayer) | (typeof(prlr) == LinearLayer)
            prlr.weight .= lr.weight
            prlr.bias .= lr.bias
        end
    end
    return predlayers
end


# ============================
# Mutable structs for layers: hold weights, bias, data storage, dims
# ============================

# avoids repeatedly creating and unstructuring very slow namedtuples
Base.@kwdef mutable struct ConvLayer
    name::Symbol = :noname
    activationf::Function = relu!
    adj::Float64 = 0.0
    weight::Array{Float64,4} = Float64[;;;;]  # (filter_h, filter_w, in_channels, out_channels)
    # weight dims
    f_h::Int64 = 0
    f_w::Int64 = 0
    inch::Int64 = 0
    outch::Int64 = 0
    bias::Vector{Float64} = Float64[]    # (out_channels)
    z::Array{Float64,4} = Float64[;;;;]
    a::Array{Float64,4} = Float64[;;;;]
    a_below::Array{Float64, 4} = Float64[;;;;]  
    # image dims
    out_h::Int64 = 0
    out_w::Int64 = 0
    in_h::Int64 = 0
    in_w::Int64 = 0
    eps_l::Array{Float64,4} = Float64[;;;;]
    pad_next_eps::Array{Float64,4} = Float64[;;;;]
    grad_a::Array{Float64,4} = Float64[;;;;]
    grad_weight::Array{Float64,4} = Float64[;;;;]
    grad_bias::Vector{Float64} = Float64[]

    b::Int64 = 0  # TODO not clear that we need this
end


Base.@kwdef mutable struct LinearLayer
    name::Symbol = :noname
    activationf::Function = relu!
    adj::Float64 = 0.0
    weight::Array{Float64,2} = Float64[;;] # (output_dim, input_dim)
    output_dim::Int64 = 0
    input_dim::Int64  = 0
    bias::Vector{Float64}  =  Float64[]     # (output_dim)
    z::Array{Float64,2} = Float64[;;]       # feed forward linear combination result 
    a::Array{Float64, 2} = Float64[;;]      # feed forward activation output
    # ALIAS TO ACTIVATION FROM LOWER BELOW used in backprop, but don't make a copy
            # type and size don't matter for an alias
            # we don't need to pre-allocate this. It just simplifies the API
            # when do we assign this?????  in layer_forward for LinearLayer
    a_below::Array{Float64, 2} = Float64[;;]  
    eps_l::Array{Float64,2} = Float64[;;]   # backprop error of the layer
    grad_a::Array{Float64,2} = Float64[;;]  # backprop derivative of activation output
    grad_weight::Array{Float64,2} = Float64[;;]
    grad_bias::Vector{Float64} = Float64[]
end


# no weight, bias, gradients, activation
Base.@kwdef mutable struct FlattenLayer
    name::Symbol = :noname
    h::Int64=0
    w::Int64=0
    ch::Int64=0
    b::Int64=0
    output_dim::Int64=0
    dl_dflat::Array{Float64,2}=Float64[;;]
    a::Array{Float64,2}=Float64[;;]
    eps_l::Array{Float64,4}=Float64[;;;;]
end


Base.@kwdef mutable struct InputLayer     # we only have this to simplify feedforward loop
    name::Symbol = :noname
    kind::Symbol = :image   # other allowed value is :linear
    out_h::Int64 = 0
    out_w::Int64 = 0
    out_c::Int64 = 0
    a::Array{Float64}   # no default provided because dims different for :image vs :linear
end


Base.@kwdef mutable struct MaxPoolLayer
    name::Symbol = :noname
    pool_size::Tuple{Int,Int}
    mask::Array{Bool,4}
    input_shape::Tuple{Int,Int,Int, Int}
end


# ============================
# pre-alllocation of Layers
# ============================
function allocate_layers(lsvec::Vector{LayerSpec}, n_samples)

    Random.seed!(42)
    layerdat = []

    for (idx,lr)  in enumerate(lsvec)
        if idx == 1
            if lr.kind != :input
                error("First layer must be the input layer.")
            else
                push!(layerdat,
                    InputLayer(
                        name = lr.name,
                        kind = lr.kind,
                        out_h = lr.h,
                        out_w = lr.w,
                        out_c = lr.outch,
                        a = zeros(lr.h, lr.w, lr.outch, n_samples),
                        ))
            end
            continue  # skip the ifs and go to next lr
        elseif lr.kind == :conv
            # filter dims
            f_h = lr.f_h
            f_w = lr.f_w
            inch = lr.inch
            outch = lr.outch
            # output image dims
            out_h=lr.h
            out_w=lr.w
            # backprop dimensions
            b_out_h = lsvec[idx-1].h   
            b_out_w = lsvec[idx-1].w   
            b_out_c = lsvec[idx-1].outch   
            # @show lr.adj
            push!(layerdat,
                ConvLayer(  
                    name = lr.name,
                    activationf =   if lr.activation ==  :relu 
                                        relu!
                                    elseif lr.activation == :none
                                        noop
                                    else error("Only :relu, and :none  supported, not $(Symbol(lr.activation)).")
                                    end,
                    adj = lr.adj,
                    weight=he_initialize((f_h,f_w,inch,outch),scale=2.0, adj=lr.adj),
                    f_h=f_h,
                    f_w=f_w,
                    inch=inch,
                    outch=outch,
                    bias=zeros(outch),
                    z=zeros(out_h, out_w, outch, n_samples),
                    a=zeros(out_h, out_w, outch, n_samples),
                    out_h = out_h,
                    out_w = out_w,
                    in_h = b_out_h,
                    in_w = b_out_w,
                    eps_l=zeros(b_out_h, b_out_w, b_out_c, n_samples),  
                    grad_a=zeros(out_h, out_w, outch, n_samples),
                    pad_next_eps=zeros(b_out_h, b_out_w, outch, n_samples),
                    grad_weight=zeros(f_h, f_w, inch, outch),
                    grad_bias=zeros(outch),
                    b=n_samples  # not clear we need this!
                    ))   

        elseif lr.kind == :linear
            # weight dims
            outputs=lr.h
            inputs = layerdat[idx-1].output_dim
            # output dims
            # @show lr.adj
            push!(layerdat,
                LinearLayer(
                    name = lr.name,
                    activationf =   if lr.activation ==  :relu 
                                        relu!
                                    elseif lr.activation == :none
                                        noop
                                    elseif lr.activation == :softmax
                                        softmax!
                                    else error("Only :relu, :softmax and :none  supported, not $(Symbol(lr.activation)).")
                                    end,
                    adj = lr.adj,
                    weight = he_initialize((outputs, inputs), scale=1.0, adj=lr.adj),
                    output_dim =  outputs,
                    input_dim =   inputs,
                    bias = zeros(outputs),
                    z=zeros(outputs, n_samples),
                    a=zeros(outputs, n_samples),
                    eps_l = zeros(outputs, n_samples),
                    grad_a = zeros(outputs, n_samples),
                    grad_weight = zeros(outputs, inputs),
                    grad_bias = zeros(outputs),
                    ))

        elseif lr.kind == :maxpool
            # TODO

        elseif lr.kind == :flatten
            h = lsvec[idx-1].h
            w = lsvec[idx-1].w
            ch = lsvec[idx-1].outch
            output_dim = h*w*ch
            b = n_samples
            push!(layerdat,
                FlattenLayer(
                    name = lr.name,
                    h=h,
                    w=w,
                    ch=ch,
                    b=b;
                    output_dim=output_dim,
                    a=zeros(output_dim, n_samples),
                    dl_dflat = zeros(output_dim, n_samples),
                    eps_l=zeros(h,w,ch,n_samples)
                    ))
        else
            error("Found unrecognized layer kind")
        end
    end

    return layerdat
end


function he_initialize(weight_dims::NTuple{4, Int64}; scale=2.0, adj=0.0)
    k_h, k_w, in_channels, out_channels = weight_dims
    fan_in = k_h * k_w * in_channels
    scale_factor = scale/(1.0 + adj^2) / Float64(fan_in)
    randn(k_h, k_w, in_channels, out_channels) .* sqrt(scale_factor)
end

function he_initialize(weight_dims::NTuple{2, Int64}; scale=2.0, adj=0.0)
    # adj is typically for leaky relu
    n_in, n_out = weight_dims
    scale_factor = scale/(1.0 + adj^2) / Float64(n_in)
    randn(n_in, n_out) .* sqrt(scale_factor)
end


# =====================
# layer functions
# =====================

# feedforward layers are all called layer_forward! and dispatch on the layer type
# backprop layers are all called layer_backward! and dispatch on the layer type

function layer_forward!(layer::ConvLayer, x::AbstractArray{Float64,4}, batch_size)
    layer.a_below = x
    # weight dims
    f_h = layer.f_h
    f_w = layer.f_w
    in_channels = layer.inch
    out_channels = layer.outch
    # input image dims
    in_h = layer.in_h
    in_w = layer.in_w
    H_out = in_h - f_h + 1  # smaller than the input h and w of x
    W_out = in_w - f_w + 1

    @inbounds for b in 1:batch_size
        for oc = 1:out_channels
            for i = 1:H_out
                for j = 1:W_out
                    for ic = 1:in_channels
                        for fh in 1:f_h
                            for fw in 1:f_w
                                layer.z[i,j,oc, b] = (x[i+fh-1, j+fw-1, ic, b] 
                                    .* layer.weight[fh,fw,ic,oc] + layer.bias[oc])
                            end
                        end
                    end
                end
            end
        end
    end
    # activation
    layer.activationf(layer, layer.adj)

    # println("Feedforward of conv outputs")
    # @show size(layer.z)
    # @show size(layer.a)
    return     # nothing
end


function layer_backward!(layer::ConvLayer, layer_next, n_samples)
    # weight (filter) dims
    f_h = layer.f_h
    f_w = layer.f_w
    inch = layer.inch
    outch = layer.outch
    # input image dims
    # output image dims
    H_out = layer.in_h
    W_out = layer.out_h
    fill!(layer.grad_weight, 0.0)  # reinitialization to allow accumulation of convolutions
    fill!(layer.eps_l, 0.0)
    fill!(layer.grad_a , 0.0)

    # println("\n***** $(layer.name)")
    # @show size(layer.eps_l)
    # @show size(layer_next.eps_l)
    # @show size(layer.pad_next_eps)
    # @show size(layer.z)
    # @show size(layer.grad_a)
        
    relu_grad!(layer.grad_a, layer.z, layer.adj)
    layer_next.eps_l .*= layer.grad_a     

    # Compute layer loss using pad_next_eps rather than the actual eps_l of the next layer, which is the wrong size
    layer.pad_next_eps .= 0.0
    @views layer.pad_next_eps[2:end-1, 2:end-1, :, :] .= layer_next.eps_l


    @inbounds for b = 1:n_samples
        for oc = 1:outch
            for i = 1:H_out-(f_h-1) # prevent filter from extending out of bounds
                for j = 1:W_out-(f_w-1)
                    for ic = 1:inch
                        for fi = 1:f_h
                            for fj = 1:f_w
                                # Flipped weight indices for backward pass: use weight[f_h-fi+1,f_w-fj+1] instead of weight[fi,fj]
                                layer.eps_l[i+fi-1,j+fj-1,ic,b] += layer.weight[f_h-fi+1,f_w-fj+1,ic,oc] * layer.pad_next_eps[i,j,oc,b] 
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
        layer.grad_bias[oc] = sum(layer.eps_l[:,:,oc,:]) .* (1.0 / Float64(n_samples))
    end

    # @show size(layer.grad_bias)
    # @show size(layer.eps_l)
    # @show size(layer.grad_a)

    return     # nothing
end

function compute_grad_weight!(layer, n_samples)
    H_out, W_out = size(layer.eps_l, 1), size(layer.eps_l, 2)
    f_h, f_w = size(layer.grad_weight, 1), size(layer.grad_weight, 2)
    batch_size = size(layer.a_below, 4)
    # @assert f_h == 3 && f_w == 3  # given 3x3 filters (for clarity)

    # Initialize grad_weight to zero
    fill!(layer.grad_weight, 0.0) # no allocations; faster than assignment

    # Use @views to avoid copying subarrays
    @inbounds for oc in axes(layer.eps_l, 3)      # 1:out_channels
        # View of the error for this output channel (all spatial positions, all batches)
        err = @view layer.eps_l[:, :, oc, :]      # size H_out × W_out × batch_size
        for ic in axes(layer.a_below, 3)          # 1:in_channels
            # View of the input activation for this channel
            # (We'll slide this view for each filter offset)
            input_chan = @view layer.a_below[:, :, ic, :]   # size H_in × W_in × batch_size
            for fi in 1:f_h
                for fj in 1:f_w
                    # Extract the overlapping region of input corresponding to eps_l[:, :, oc, :]
                    local_patch = @view input_chan[fi:fi+H_out-1, fj:fj+W_out-1, :]
                    # Accumulate gradient for weight at (fi,fj, ic, oc)
                    layer.grad_weight[fi, fj, ic, oc] += sum(local_patch .* err)
                end
            end
        end
    end

    # Average over batch (divide by batch_size)
    layer.grad_weight .*= (1 / n_samples)
    return   # nothing
end

function layer_forward!(layer::MaxPoolLayer, x::Array{Float64,4})
    layer.input_shape = size(x)
    (pool_h, pool_w) = layer.pool_size
    (H, W, C, B) = size(x)
    H_out = div(H, pool_h)
    W_out = div(W, pool_w)
    y = zeros(Float64, H_out, W_out, C, B)
    layer.mask = falses(size(x))
    for bn = 1:B
        for c = 1:C
            for i = 1:H_out
                for j = 1:W_out
                    region = view(x, (pool_h*(i-1)+1):(pool_h*i), (pool_w*(j-1)+1):(pool_w*j), c, bn)
                    max_val = maximum(region)
                    y[i,j,c, bn] = max_val
                    for a = 1:pool_h, b = 1:pool_w
                        if region[a,b] == max_val && !layer.mask[pool_h*(i-1)+a, pool_w*(j-1)+b, c, bn]
                            layer.mask[pool_h*(i-1)+a, pool_w*(j-1)+b, c, bn] = true
                            break
                        end
                    end
                end
            end
        end
    end
    return y
end

function layer_backward!(layer::MaxPoolLayer, d_out::Array{Float64,4})
    (H, W, C, B) = layer.input_shape
    d_x = zeros(Float64, H, W, C, B)
    (pool_h, pool_w) = layer.pool_size
    (H_out, W_out, C_out, B) = size(d_out)
    for bn = 1:B
        for c = 1:C
            for i = 1:H_out
                for j = 1:W_out
                    for a = 1:pool_h, b = 1:pool_w
                        if layer.mask[pool_h*(i-1)+a, pool_w*(j-1)+b, c, bn]
                            d_x[pool_h*(i-1)+a, pool_w*(j-1)+b, c, bn] = d_out[i,j,c, bn]
                        end
                    end
                end
            end
        end
    end
    return d_x
end


# TODO fix this with pre-allocated storage
function layer_forward!(layer::FlattenLayer, x::Array{Float64, 4}, batch_size) 
    layer.a .= reshape(x,:,layer.b)
    return     # nothing
end


function layer_backward!(layer::FlattenLayer, layer_next::LinearLayer, batch_size)
    # @show size(layer.dl_dflat)
    # @show size(layer_next.weight')
    # @show size(layer_next.eps_l)
    layer.dl_dflat = layer_next.weight' * layer_next.eps_l  # TODO element-wise times current layer's relu'
    layer.eps_l .= reshape(layer.dl_dflat,layer.h, layer.w, layer.ch, :)
    return     # nothing
end


function layer_forward!(layer::LinearLayer, x::Matrix{Float64}, batch_size)
    layer.z .= layer.weight * x .+ layer.bias  # TODO test this for allocations
    layer.a_below = x   # assign alias for using in backprop
    # activation
    layer.activationf(layer, layer.adj)
    return     # nothing
end

function layer_backward!(layer::LinearLayer, layer_next::LinearLayer, n_samples; output=false)
    if output
        # layer.eps_l calculated by prior call to dloss_dz
        layer.grad_weight .= (layer.eps_l * layer.a_below') .* (1.0 / n_samples)  # this x is activation of lower layer
    else  # this is hidden layer
        relu_grad!(layer.grad_a, layer.z, layer.adj)  # calculates layer.grad_a
        layer.eps_l .= (layer_next.weight' * layer_next.eps_l) .* layer.grad_a   
        layer.grad_weight .= (layer.eps_l * layer.a_below') .* (1.0 / n_samples)  # this x is activation of lower layer
    end
    layer.grad_bias .= sum(layer.eps_l, dims=2) .* (1.0 / n_samples)  # sum(epsilon, dims=2) .* (1.0 / n)
    # @show size(layer.eps_l)
    # @show size(layer.bias)
    # @show size(layer.grad_weight)
    return     # nothing
end


# this can be used in a special position in the backprop! function and 
# will dispatch properly because it's the only method with one input
function dloss_dz!(layer, target)
    layer.eps_l .= layer.a .- target 
end


function relu!(layer, adj)
    @fastmath layer.a .= max.(layer.z, adj)
end

# use for activation of conv or linear, when activation is requested as :none
function noop(args...)
end

function relu_grad!(grad, z, adj)   # I suppose this is really leaky_relu...
    @inbounds for i = eachindex(z)  # when passed any array, this will update in place
        grad = ifelse(z[i] > 0.0, 1.0, adj)  # prevent vanishing gradients by not using 0.0
    end
end

# no allocations here, mom!
function softmax!(layer, adj=0.0) # adj arg required for calling loop: not used
    for c in axes(layer.z,2)
        va = view(layer.a, :, c)
        vz = view(layer.z, :, c)
        va .= exp.(vz .- maximum(vz))
        va .= va ./ (sum(va) .+ 1e-12)
    end
    return
end


function accuracy(preds, targets)  # this is NOT very general
    if size(targets,1) > 1
        # targetmax = ind2sub(size(targets),vec(findmax(targets,1)[2]))[1]
        # predmax = ind2sub(size(preds),vec(findmax(preds,1)[2]))[1]
        targetmax = getvalidx(targets)     # vec(map(x -> x[1], argmax(targets,dims=1)));
        predmax =   getvalidx(preds)       # vec(map(x -> x[1], argmax(preds,dims=1)));
        fracright = mean(targetmax .== predmax)
    else
        # works because single output unit is classification probability
        # choices = [j > 0.5 ? 1.0 : 0.0 for j in preds]
        choices = zeros(size(preds))
        for i = eachindex(choices)
            choices[i] = preds[i] > 0.5 ? 1.0 : 0.0
        end
        fracright = mean(choices .== targets)
    end
    return fracright
end


function getvalidx(arr, argfunc=argmax)  # could also be argmin
    return vec(map(x -> x[1], argfunc(arr, dims=1)))
end

function cross_entropy_loss(pred::AbstractMatrix{Float64}, target::AbstractMatrix{Float64}, n_samples)
    # target is one-hot encoded
    # return -sum(target .* log.(pred .+ 1e-9))
    n = n_samples
    cost = (-1.0 / n) * (dot(target,log.(max.(pred, 1e-20))) +
        dot((1.0 .- target), log.(max.(1.0 .- pred, 1e-20))))
end


# ============================
# Training Loop
# ============================

function feedforward!(layers, x_train, n_samples)
    layers[begin].a = x_train
    @inbounds for (i,lr) in enumerate(layers[2:end])  # assumes that layers[1] MUST be input layer without checking!
        i += 1  # skip index of layer 1
        # dispatch on type of lr
        # activation function is part of the layer definition
        layer_forward!(lr, layers[i-1].a, n_samples) 
    end
end

function backprop!(layers, y_train, n_samples)
    # output layer is different
    dloss_dz!(layers[end], y_train)  
    layer_backward!(layers[end],layers[end-1], n_samples; output=true)

    # skip over output layer (end) and input layer (begin)
    @inbounds for (j, lr) in enumerate(reverse(layers[begin+1:end-1])); 
        i = length(layers) - j; 
        # @show i, typeof(lr)
        layer_backward!(lr, layers[i+1], n_samples)
    end
    return
end

function weight_update!(layer, lr)
    layer.weight .-= lr .* layer.grad_weight
    layer.bias .-= lr .* layer.grad_bias
end

function update_weight_loop!(layers, lambda)
    for lr in layers[begin+1:end]
        if (typeof(lr) == FlattenLayer) | (typeof(lr) == MaxPoolLayer)
            continue  # redundant but obvious
        end
        weight_update!(lr, lambda)
    end
end


function train_loop!(layers; x_train, y_train, batch_size, epochs, minibatch_size=0, lr=0.01)

    # setup minibatches
    if minibatch_size == 0
        # setup noop minibatch parameters
        mini_num = 1
        n_samples = batch_size
    else
        if rem(batch_size, minibatch_size) != 0
            error("minibatch_size does not divide evenly into batch_size")
        else
            mini_num = div(batch_size,  minibatch_size)
            n_samples = minibatch_size
        end
    end

    @show mini_num
    @show n_samples

    for e = 1:epochs
        @inbounds for batno in 1:mini_num
            if mini_num > 1
                start_obs = (batno - 1) * minibatch_size + 1
                end_obs = start_obs + minibatch_size - 1
                x_train_part = view(x_train, :,:,:, start_obs:end_obs)
                y_train_part = view(y_train, :, start_obs:end_obs)
            else
                x_train_part = x_train
                y_train_part = y_train
            end

            # @show size(x_train_part)
            # @show size(y_train_part)

            newfeedforward!(layers, x_train_part, n_samples)

            newbackprop!(layers, y_train_part, n_samples)

            update_weight_loop!(layers, lr)

            loss_val = cross_entropy_loss(layers[end].a, y_train_part, n_samples)
            acc = accuracy(layers[end].a, y_train_part)
            println("\nepoch $e batch $batno Loss = $loss_val Accuracy = $acc\n")
        end
    end
end


function predict(predlayers, x_input, y_input)
    n_samples = size(x_input, ndims(x_input))
    newfeedforward!(predlayers, x_input, n_samples)
    preds = predlayers[end].a
    acc = accuracy(preds, y_input)
    cost = cross_entropy_loss(preds, y_input, n_samples)
    println("Accuracy $acc  Cost $cost")
end


# ============================
# Utility Functions
# ============================

function display_mnist_digit(digit_data, dims=[])
    if length(dims) == 0
        xside = yside = convert(Int,(sqrt(length(digit_data))))
    elseif length(dims) == 1
        xside = yside = dims[1]
    elseif length(dims) >= 2
        xside = dims[2]
        yside = dims[1]
    end
    plot(Gray.(transpose(reshape(digit_data,xside,yside))),  interpolation="nearest", showaxis=false) 
    # println("Press enter to close image window..."); readline()
    # close("all")        
end

function random_onehot(i,j)
    arr = zeros(i,j)
    for n in axes(arr,2)
        rowselector = rand(1:10)
        arr[rowselector, n] = 1.0
    end
    return arr
end

# ============================
# Save and load weights to/from files
# ============================

function weights2file(layers, suffix, pathstr)
    for lr in fieldnames(typeof(layers))
        serialize(joinpath(pathstr,string(lr)*"_bias"*'_'*suffix*".dat"), 
                            eval(getfield(layers,lr)).bias)
        serialize(joinpath(pathstr, string(lr)*"_weight"*'_'*suffix*".dat"), 
                            eval(getfield(layers,lr)).weight)
    end
end

function file2weights(suffix, pathstr)
    outlayers = init_layers(n_samples=batch_size)
    for lr in fieldnames(typeof(outlayers))
        fname_bias = joinpath(pathstr,string(lr)*"_bias"*'_'*suffix*".dat")
        fname_weight = joinpath(pathstr, string(lr)*"_weight"*'_'*suffix*".dat")
        setfield!(getfield(layers, lr), :bias, deserialize(fname_bias))
        setfield!(getfield(layers, lr), :weight, deserialize(fname_weight))
    end
    return outlayers
end
