#=
TODO
- test ConvLayer with no padding
- why don't we do implicit flatten when conv precedes linear?

- pre-allocate output layer for softmax
=#

using Random
using LinearAlgebra
using Colors, Plots
using Serialization
using MLDatasets
using StatsBase


# ============================
# Setup training
# ============================
"""
    struct LayerSpec

Provide input parameters to define a layer of a neural network model.
These parameters are the union of all parameters needed across many
types of layers. There are convenience methods that limit the inputs
to only the parameters needed for a specific type of layer.

These inputs are used to fully define each type of layer including
the weights and arrays required during model training.
"""
@Base.kwdef struct LayerSpec
    name::Symbol=:noname
    kind::Symbol=:none
    activation::Symbol=:none
    adj::Float64=0      # leaky_relu factor. also for he_initialize
    h::Int64=0          # image height (rows) or output neurons for linear layers
    w::Int64=0          # image width (columns)
    outch::Int64=0
    f_h::Int64=0        # filter height (rows)
    f_w::Int64=0        # filter width (columns)
    inch::Int64=0
    padrule::Symbol=:same       # either :same or :none
    stride::Int64=1             # no input required to accept default
end

# LayerSpec methods for specific kinds of layers
"""
    convlayerspec(;name::Symbol, activation::Symbol, adj::Float64=0.002, h::Int64=0, w::Int64=0, outch::Int64=0, f_h::Int64, f_w::Int64, inch::Int64=0, padrule::Symbol=:same)

Only inputs needed for a convlayer are passed to the LayerSpec. 
Note that h, w, and inch will be calculated from the previous layer,
which should be an image input, another conv layer, or a maxpooling layer.
You must provide inputs for name, activation, outch, f_h, and f_w.
"""
function convlayerspec(;name::Symbol, activation::Symbol=:relu, adj::Float64=0.002, h::Int64=0, w::Int64=0, outch::Int64, f_h::Int64, f_w::Int64, inch::Int64=0, padrule::Symbol=:same)
    LayerSpec(name=name, kind=:conv, activation=activation, adj=adj, h=h, w=w, outch=outch,f_h=f_h, f_w=f_w, inch=inch, padrule=padrule )
end

function linearlayerspec(;name::Symbol, activation::Symbol=:relu, adj::Float64=0.002, output::Int64)
    LayerSpec(name=name, kind=:linear, activation=activation, adj=adj, h=output)
end

function maxpoollayerspec(;name::Symbol, f_h::Int, f_w::Int)
    LayerSpec(name=name, kind=:maxpool, f_h=f_h, f_w=f_w)
end

function flattenlayerspec(;name::Symbol)
    LayerSpec(name=name, kind=:flatten)
end

# ============================
# Sample model definitions or modelspecs
# ============================

big_conv = LayerSpec[
    LayerSpec(h=28, w=28, outch=10, kind=:input, name=:input)
    LayerSpec(h=26, w=26, outch=4, f_h=3, f_w=3, inch=1, kind=:conv, name=:conv1, activation=:relu, adj=0.002)
    LayerSpec(h=24, w=24, outch=4, f_h=3, f_w=3, inch=4, kind=:conv, name=:conv2, activation=:relu, adj=0.002)
    LayerSpec(h=24, w=24, outch=4, kind=:flatten, name=:flatten)
    LayerSpec(h=500, kind=:linear, activation=:relu, name=:linear3, adj=0.002)
    LayerSpec(h=100, kind=:linear, activation=:relu, name=:linear4, adj=0.002)
    LayerSpec(h=10, kind=:linear,activation=:softmax, name=:output)
    ]

small_conv = LayerSpec[
        LayerSpec(name=:input, h=28, w=28, outch=1, kind=:input, )
        convlayerspec(name=:conv1, outch=32, f_h=3, f_w=3, activation=:relu, adj=0.0)
        maxpoollayerspec(name=:maxpool1, f_h=2, f_w=2)
        # convlayerspec(name=:conv2, outch=48, f_h=3, f_w=3, activation=:relu, adj=0.0)
        # maxpoollayerspec(name=:maxpool2, f_h=2, f_w=2)
        flattenlayerspec(name=:flatten)
        linearlayerspec(name=:linear1, output=128, adj=0.0)
        # linearlayerspec(name=:linear2, output=64, adj=0.0)
        LayerSpec(name=:output, h=10, kind=:linear, activation=:softmax, )
        ]

little_conv = LayerSpec[
        LayerSpec(h=28, w=28, outch=1, kind=:input, name=:input)
        convlayerspec(outch=32, f_h=3, f_w=3, name=:conv1, activation=:relu, adj=0.0)
        LayerSpec(f_h=2, f_w=2, kind=:maxpool, name=:maxpool2)
        LayerSpec(kind=:flatten, name=:flatten)
        LayerSpec(h=128, kind=:linear, activation=:relu, name=:linear1, adj=0.0)
        LayerSpec(h=64, kind=:linear, activation=:relu, name=:linear2, adj=0.0)
        LayerSpec(h=10, kind=:linear, activation=:softmax, name=:output)
        ]

two_linear = LayerSpec[
    LayerSpec(h=28, w=28, outch=1, kind=:input, name=:input)
    flattenlayerspec(name=:flatten)
    linearlayerspec(name=:linear1, output=200, adj=0.01)
    linearlayerspec(name=:linear2, output=200, adj=0.01)
    LayerSpec(h=10, kind=:linear, name=:output, activation=:softmax)
    ]



# method to pass a function
function preptrain(modelspecs::Function, batch_size, minibatch_size)
    layerspecs = modelspecs() 
    preptrain(layerspecs, batch_size, minibatch_size)
end

# method to pass a vector
function preptrain(layerspecs::Vector{LayerSpec}, batch_size, minibatch_size)
    trainset = MNIST(:train)
    testset = MNIST(:test)

    # batch_size = 10000
    # minibatch_size = 50
    x_train = trainset.features[1:28, 1:28,1:batch_size]
    x_train = Float64.(x_train)
    x_train = reshape(x_train, 28, 28, 1, batch_size)

    y_train = trainset.targets[1:batch_size]
    y_train = indicatormat(y_train)
    y_train = Float64.(y_train)

    # shuffle the variables and outcome identically
    img_idx = shuffle(1:size(x_train,4))
    x_train_shuf = x_train[:,:,:,img_idx]
    y_train_shuf = y_train[:,img_idx]

    layers = allocate_layers(layerspecs, minibatch_size);
    show_all_array_sizes(layers)

    return layerspecs, layers, x_train_shuf, y_train_shuf
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
    padrule::Symbol = :same   # other option is :none
    stride::Int64 = 1     # assume stride is symmetrical for now
    bias::Vector{Float64} = Float64[]    # (out_channels)
    z::Array{Float64,4} = Float64[;;;;]
    pad_x::Array{Float64, 4} = Float64[;;;;]
    a::Array{Float64,4} = Float64[;;;;]
    a_below::Array{Float64, 4} = Float64[;;;;]  
    pad_a_below::Array{Float64, 4} = Float64[;;;;]
    eps_l::Array{Float64,4} = Float64[;;;;]
    pad_next_eps::Array{Float64,4} = Float64[;;;;]  # TODO need to test if this is needed given successive conv layer sizes
    grad_a::Array{Float64,4} = Float64[;;;;]
    grad_weight::Array{Float64,4} = Float64[;;;;]
    grad_bias::Vector{Float64} = Float64[]
end

# method to do prep calculations based on LayerSpec inputs, then create a LinearLayer
function ConvLayer(lr::LayerSpec, prevlayer, n_samples)
    outch = lr.outch
    prev_h, prev_w, inch, _ = size(prevlayer.a)

    pad = ifelse(lr.padrule == :same, 1, 0)
    # output image dims: calculated once rather than over and over in training loop
    out_h = div((prev_h + 2pad - lr.f_h), lr.stride) + 1
    out_w = div((prev_w + 2pad - lr.f_w), lr.stride) + 1
    ConvLayer(
        name = lr.name,
        activationf =   if lr.activation ==  :relu 
                            relu!
                        elseif lr.activation == :none
                            noop
                        else error("Only :relu, and :none  supported, not $(Symbol(lr.activation)).")
                        end,
        adj = lr.adj,
        weight=he_initialize((lr.f_h, lr.f_w, inch, lr.outch), scale=2.2, adj=lr.adj),
        padrule=lr.padrule,   
        stride=lr.stride,
        bias=zeros(outch),
        pad_x = zeros(out_h + 2pad, out_w + 2pad, inch, n_samples),
        pad_a_below = zeros(prev_h + 2pad, prev_w + 2pad, inch, n_samples),
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

# method to do prep calculations based on LayerSpec inputs, then create a LinearLayer
function LinearLayer(lr::LayerSpec, prevlayer, n_samples)
    # weight dims
    outputs=lr.h        # rows
    inputs=prevlayer.output_dim    # columns
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
        weight = he_initialize((outputs, inputs), scale=1.5, adj=lr.adj),
        output_dim =  outputs,
        input_dim =   inputs,
        bias = zeros(outputs),
        z=zeros(outputs, n_samples),
        a=zeros(outputs, n_samples),
        eps_l = zeros(outputs, n_samples),
        grad_a = zeros(outputs, n_samples),
        grad_weight = zeros(outputs, inputs),
        grad_bias = zeros(outputs))  
end

# no weight, bias, gradients, activation
Base.@kwdef mutable struct FlattenLayer
    name::Symbol = :noname
    output_dim::Int64=0
    dl_dflat::Array{Float64,2}=Float64[;;]
    a::Array{Float64,2}=Float64[;;]
    eps_l::Array{Float64,4}=Float64[;;;;]
end

# method to prepare inputs and create layer
function FlattenLayer(lr::LayerSpec, prevlayer, n_samples)
    h, w, ch, _ = size(prevlayer.a)
    output_dim = h * w * ch

    FlattenLayer(
        name = lr.name,
        output_dim=output_dim,
        a=zeros(output_dim, n_samples),
        dl_dflat = zeros(output_dim, n_samples),
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
    a::Array{Float64, 4} = Float64[;;;;]
    mask::Array{Bool,4} = Bool[;;;;]
    eps_l::Array{Float64, 4} = Float64[;;;;]
end

# method to prepare inputs and create layer
function MaxPoolLayer(lr::LayerSpec, prevlayer, n_samples)

    in_h, in_w, outch, _ = size(prevlayer.grad_a)
    out_h = div(in_h, lr.f_h) # assume stride = lr.f_h implicit in code
    out_w = div(in_w,lr.f_w)  # ditto
    batch_size=n_samples

    MaxPoolLayer(
        name  = lr.name,
        pool_size = (lr.f_h, lr.f_w),
        a = zeros(out_h, out_w, outch, batch_size),
        mask = falses(in_h, in_w, outch, batch_size),
        eps_l = zeros(in_h, in_w, outch, batch_size),
        )
end

Base.@kwdef mutable struct stat_series
    acc::Array{Float64, 1} = Float64[]
    loss::Array{Float64, 1} = Float[]
    batch_size::Int = 0
    epochs::Int = 0
    minibatch_size = 0
end


function show_array_sizes(layer::InputLayer)
    @show layer.name
    @show size(layer.a)
    return
end

function show_array_sizes(layer::ConvLayer)
    @show layer.name
    @show size(layer.z)
    @show size(layer.pad_x)
    @show size(layer.a)
    @show size(layer.a_below)
    @show size(layer.pad_a_below)
    @show size(layer.eps_l)
    @show size(layer.pad_next_eps)
    @show size(layer.grad_a)
    @show size(layer.grad_weight)
    @show size(layer.grad_bias)
    return
end

function show_array_sizes(layer::FlattenLayer)
    @show layer.name
    @show layer.output_dim
    @show size(layer.a)
    @show size(layer.eps_l)
    return
end

function show_array_sizes(layer::MaxPoolLayer)
    @show layer.name
    @show layer.pool_size
    # @show input_shape
    @show size(layer.a)
    @show size(layer.mask)
    @show size(layer.eps_l)
    return
end

function show_array_sizes(layer::LinearLayer)
    @show layer.name
    @show size(layer.weight)
    @show layer.output_dim
    @show layer.input_dim
    @show size(layer.bias)
    @show size(layer.z)
    @show size(layer.a)
    @show size(layer.a_below)
    @show size(layer.eps_l)
    @show size(layer.grad_a)
    @show size(layer.grad_weight)
    @show size(layer.grad_bias)
    return
end

function show_all_array_sizes(layers)
    for lr in layers
        println()
        show_array_sizes(lr)
        println()
    end
    return
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
                        outch = lr.outch,
                        a = zeros(lr.h, lr.w, lr.outch, n_samples),
                        ))
            end
            continue  # skip the ifs and go to next lr
        elseif lr.kind == :conv
            push!(layerdat, ConvLayer(lr, layerdat[idx-1], n_samples))
        elseif lr.kind == :linear
            push!(layerdat, LinearLayer(lr, layerdat[idx-1], n_samples))
        elseif lr.kind == :maxpool
            push!(layerdat, MaxPoolLayer(lr, layerdat[idx-1], n_samples))
        elseif lr.kind == :flatten
            push!(layerdat, FlattenLayer(lr, layerdat[idx-1], n_samples))
        else
            error("Found unrecognized layer kind")
        end
    end

    return layerdat
end

function allocate_stats(batch_size, minibatch_size, epochs)
    no_of_batches = div(batch_size, minibatch_size)
    stats = stat_series(
            acc = zeros(no_of_batches * epochs),
            loss = zeros(no_of_batches * epochs),
            batch_size=batch_size,  
            epochs=epochs,
            minibatch_size=minibatch_size)
end


# ============================
# Setup prediction
# ============================

function setup_preds(modelspecs::Function, layers, n_samples)
    predlayerspecs = modelspecs()
    setup_preds(predlayerspecs, layers, n_samples)
end


function setup_preds(predlayerspecs, layers, n_samples)
    predlayers = allocate_layers(predlayerspecs, n_samples)
    for (prlr, lr) in zip(predlayers, layers)
        if (typeof(prlr) == ConvLayer) | (typeof(prlr) == LinearLayer)
            prlr.weight .= lr.weight
            prlr.bias .= lr.bias
        end
    end
    return predlayers
end


# =====================
# layer functions
# =====================

# feedforward layers are all called layer_forward! and dispatch on the layer type
# backprop layers are all called layer_backward! and dispatch on the layer type

function layer_forward!(layer::ConvLayer, x::AbstractArray{Float64,4}, batch_size)
    layer.a_below = x   # as an alias, this might not work for backprop though it seems to

    if layer.padrule == :same  # we know padding = 1 for :same
        @views layer.pad_x[begin+1:end-1, begin+1:end-1, :, :] .= x 
    end

    @inbounds for b in axes(layer.z, 4)
        for oc in axes(layer.z, 3)  # this could also be axes(layer.weight, 4)
            for j in axes(layer.z, 2)
                for i in axes(layer.z, 1)
                    for ic in axes(layer.weight, 3)
                        for fw in axes(layer.weight, 2)
                            for fh in axes(layer.weight, 1)
                                layer.z[i,j,oc, b] = (layer.pad_x[i+fh-1, j+fw-1, ic, b] 
                                    * layer.weight[fh,fw,ic,oc] + layer.bias[oc])
                            end
                        end
                    end
                end
            end
        end
    end
    # activation
    layer.activationf(layer, layer.adj)

    return     # nothing
end


function layer_backward!(layer::ConvLayer, layer_next, n_samples)
    f_h, f_w, _, _ = size(layer.weight)
    H_out, W_out, _, _ = size(layer.eps_l)

    fill!(layer.grad_weight, 0.0)  # reinitialization to allow accumulation of convolutions
    fill!(layer.eps_l, 0.0)
    fill!(layer.grad_a , 0.0)
        
    relu_grad!(layer.grad_a, layer.z, layer.adj)

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


    # @inbounds for b = 1:n_samples  # -> axes(eps_l, 4)
    @inbounds for b in axes(layer.eps_l, 4)
        for oc in axes(layer.weight, 4)
            for i = 1:H_out-(f_h-1) # prevent filter from extending out of bounds
                for j = 1:W_out-(f_w-1)  
                    for ic in axes(layer.weight, 3)
                        for fj in axes(layer.weight,2)
                            for fi in axes(layer.weight,1)
                                # Flipped weight indices for backward pass: use weight[f_h-fi+1,f_w-fj+1] instead of weight[fi,fj]
                                layer.eps_l[i+fi-1,j+fj-1,ic,b] += layer.weight[f_h-fi+1, f_w-fj+1, ic, oc] * layer.pad_next_eps[i,j,oc,b] 
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

    return     # nothing
end

function compute_grad_weight!(layer, n_samples)
    H_out, W_out, _, _ = size(layer.eps_l)

    # Initialize grad_weight to zero
    fill!(layer.grad_weight, 0.0) # no allocations; faster than assignment
    if layer.padrule == :same
        fill!(layer.pad_a_below, 0.0)
        layer.pad_a_below[2:end-1, 2:end-1, :, :] .= layer.a_below
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
            for fj in axes(layer.weight,2)
                for fi in axes(layer.weight,1)
                    # Extract the overlapping region of input corresponding to eps_l[:, :, oc, :]
                    local_patch = @view input_chan[fi:fi+H_out-1, fj:fj+W_out-1, :]   
                    # Accumulate gradient for weight at (fi,fj, ic, oc)
                    # layer.grad_weight[fi, fj, ic, oc] += sum(local_patch .* err)
                    layer.grad_weight[fi, fj, ic, oc] += dot(err, local_patch)
                end 
            end
        end
    end

    # Average over batch (divide by batch_size)
    layer.grad_weight .*= (1 / n_samples)
    return   # nothing
end

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
                    layer.a[i,j,c,bn] = max_val
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
    return  # nothing
end


function layer_backward!(layer::MaxPoolLayer, layer_next, n_samples)
    fill!(layer.eps_l, 0.0)
    (pool_h, pool_w) = layer.pool_size
    @inbounds for bn in axes(layer_next.eps_l,4)
        for c in axes(layer.eps_l,3)
            for j = axes(layer_next.eps_l, 2)  #  1:W_in
                for i =  axes(layer_next.eps_l, 1) #  1:H_in
                    for b = 1:pool_w, a = 1:pool_h   # gives 2 rows and 2 columns for each i, j
                        if layer.mask[pool_h*(i-1)+a, pool_w*(j-1)+b, c, bn]
                            layer.eps_l[pool_h*(i-1)+a, pool_w*(j-1)+b, c, bn] = layer_next.eps_l[i,j,c, bn]
                            break
                        end
                    end
                end
            end
        end
    end
    return    # nothing
end


function layer_forward!(layer::FlattenLayer, x::Array{Float64, 4}, batch_size) 
    layer.a .= reshape(x,:,batch_size)  
    return     # nothing
end


function layer_backward!(layer::FlattenLayer, layer_next::LinearLayer, batch_size)
    # Use pre-allocated dl_dflat array for matrix multiplication
    mul!(layer.dl_dflat, layer_next.weight', layer_next.eps_l)  # in-place matrix multiplication
    # Reshape in-place using views
    h, w, ch, _ = size(layer.eps_l)
    @inbounds for i in 1:batch_size
        layer.eps_l[:,:,:,i] .= reshape(@view(layer.dl_dflat[:,i]), h, w, ch)
    end
    return     
end


function layer_forward!(layer::LinearLayer, x::Matrix{Float64}, batch_size)
    mul!(layer.z, layer.weight, x)  # in-place matrix multiplication
    layer.z .+= layer.bias  # in-place addition with broadcasting
    layer.a_below = x   # assign alias for using in backprop
    # activation
    layer.activationf(layer, layer.adj)
    return     
end

function layer_backward!(layer::LinearLayer, layer_next::LinearLayer, n_samples; output=false)
    if output
        # layer.eps_l calculated by prior call to dloss_dz
        mul!(layer.grad_weight, layer.eps_l, layer.a_below')  # in-place matrix multiplication
        layer.grad_weight .*= (1.0 / n_samples)  # in-place scaling
    else  # this is hidden layer
        relu_grad!(layer.grad_a, layer.z, layer.adj)  # calculates layer.grad_a
        mul!(layer.eps_l, layer_next.weight', layer_next.eps_l)  # in-place matrix multiplication
        layer.eps_l .*= layer.grad_a  # in-place element-wise multiplication
        mul!(layer.grad_weight, layer.eps_l, layer.a_below')  # in-place matrix multiplication
        layer.grad_weight .*= (1.0 / n_samples)  # in-place scaling
    end
    # Compute bias gradient efficiently without allocations
    fill!(layer.grad_bias, 0.0)  # Reset to zero
    for j in axes(layer.eps_l, 2)  # Iterate over columns (batch dimension)
        for i in axes(layer.eps_l, 1)  # Iterate over rows (output dimension)
            layer.grad_bias[i] += layer.eps_l[i,j]
        end
    end
    layer.grad_bias .*= (1.0 / n_samples)  # in-place scaling
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
        grad[i] = ifelse(z[i] > 0.0, 1.0, adj)  # prevent vanishing gradients by not using 0.0
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

    stats = allocate_stats(batch_size, minibatch_size, epochs)
    counter = 0

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
            counter += 1

            @show counter
            feedforward!(layers, x_train_part, n_samples)

            backprop!(layers, y_train_part, n_samples)

            update_weight_loop!(layers, lr)

            gather_stats!(stats, layers, y_train_part, counter, batno, e; to_console=false)

        end
    end
    return stats
end


function gather_stats!(stat_series, layers,  y_train, counter, batno, epoch; to_console=true, to_series=true)
    loss_val = cross_entropy_loss(layers[end].a, y_train, stat_series.minibatch_size)
    acc = accuracy(layers[end].a, y_train)
    if to_series
        stat_series.loss[counter] = loss_val
        stat_series.acc[counter] = acc
    end

    if to_console
        println("\nepoch $epoch batch $batno Loss = $loss_val Accuracy = $acc\n")
    end
end

function predict(predlayers, x_input, y_input)
    n_samples = size(x_input, ndims(x_input))
    feedforward!(predlayers, x_input, n_samples)
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

function plot_training(stats)
    plot(stats.acc, label="Accuracy", color=:blue, ylabel="Accuracy")  
    plot!(twinx(), stats.loss, label="Cost", ylabel="Loss",color=:red)
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
