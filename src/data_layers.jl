using LoopVectorization

# ============================
# abstract types for layer and normparams
# forward declaration to permit these to
# be used when defining data layers
# of the neural network.
# ============================
abstract type Layer end     # super type for all layers
abstract type NormParam end # super type for all normalization parameters
abstract type OptParam end  # super type for Optimization params and arrays


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
    name::Symbol
    kind::Symbol
    isoutput::Bool = false
    activation::Symbol = :none      # options are :relu, :leaky_relu, :logistic;  for output layer only :softmax, :regression
    normalization::Symbol = :none   # options are :none, :batchnorm
    optimization::Symbol = :none    # options are :none :adam :adamw
    adj::ELT = 0                    # leaky_relu factor. also for he_initialize
    h::Int64 = 0                    # image height (rows) for conv, flatten, maxpool
    w::Int64 = 0                    # image width (columns) for conv, flatten, maxpool
    outch::Int64 = 0                # output channels in image format array
    outputdim::Int64 = 0            # output neurons for linear layers
    f_h::Int64 = 0                  # filter height (rows)
    f_w::Int64 = 0                  # filter width (columns)
    padrule::Symbol = :same         # either :same or :none
    stride::Int64 = 1               # no input required to accept default
end

# LayerSpec methods for specific kinds of layers
"""
    convlayerspec(;name::Symbol, isoutput::Bool, activation::Symbol, adj::ELT=0.002, h::Int64=0, w::Int64=0, outch::Int64=0, f_h::Int64, f_w::Int64, padrule::Symbol=:same)

Only inputs needed for a convlayer are passed to the LayerSpec.
Note that h, w, and inch will be calculated from the previous layer,
which should be an image input, another conv layer, or a maxpooling layer.
You must provide inputs for name, activation, outch, f_h, and f_w.
"""
function convlayerspec(; name::Symbol, activation::Symbol=:relu, normalization::Symbol=:none, optimization::Symbol=:none, isoutput=false,
    adj::ELT=ELT(0.002), outch::Int64, f_h::Int64, f_w::Int64, padrule::Symbol=:same)
    LayerSpec(name=name, kind=:conv, activation=activation, normalization=normalization, optimization=optimization, isoutput=isoutput,
        adj=adj, outch=outch, f_h=f_h, f_w=f_w, padrule=padrule)
end

function linearlayerspec(; name::Symbol, activation::Symbol=:relu, normalization::Symbol=:none, optimization::Symbol=:none,
    isoutput=false, adj::ELT=ELT(0.002), outputdim::Int64)
    LayerSpec(name=name, kind=:linear, isoutput=isoutput, activation=activation, normalization=normalization, optimization=optimization, adj=adj, outputdim=outputdim)
end

function maxpoollayerspec(; name::Symbol, f_h::Int, f_w::Int)
    LayerSpec(name=name, kind=:maxpool, f_h=f_h, f_w=f_w)
end

function flattenlayerspec(; name::Symbol)
    LayerSpec(name=name, kind=:flatten)
end

"""
    inputlayerspec(name=:input_image, h=28, w=28, outch=1)
Supply inputs for h, w,and outch for an input array that is an image stack with dimensions of h, w, outch, batchsize.

    inputlayerspec(name=:input_array, outputdim=15)
Or supply only an input for the output parameter for a dense linear input array of dimensions output, batchsize.

For both array inputs, the batchsize will be determined from the actual array provided as x input to function trainloop!.
"""
function inputlayerspec(; name::Symbol, h::Int64=0, w::Int64=0, outch::Int64=0, outputdim::Int64=0)
    if (h > 0) & (w > 0) & (outch > 0)
        LayerSpec(name=name, kind=:input, h=h, w=w, outch=outch)
    elseif outputdim > 0
        LayerSpec(name=name, kind=:input, outputdim=outputdim)
    else
        error("Positive integer inputs must be supplied for h, w, AND outch OR only for output.")
    end
end


"""
    outputlayerspec(; name::Symbol, activation::Symbol, optimization::Symbol=:none, outputdim::Int64)

Create a LayerSpec for an output layer with specified activation function.

# Arguments
- `name::Symbol`: Name of the output layer
- `activation::Symbol`: Output activation function - must be one of `:softmax`, `:logistic`, or `:regression`
- `optimization::Symbol=:none`: Optimization method (`:none`, `:adam`, or `:adamw`)
- `outputdim::Int64`: Number of output units

# Examples
```julia
# For multiclass classification (10 classes)
outputlayerspec(name=:output, activation=:softmax, outputdim=10)

# For binary classification
outputlayerspec(name=:output, activation=:logistic, outputdim=1)

# For regression
outputlayerspec(name=:output, activation=:regression, outputdim=1)

# With optimization
outputlayerspec(name=:output, activation=:softmax, optimization=:adam, outputdim=10)
```
"""
function outputlayerspec(; name::Symbol, activation::Symbol, optimization::Symbol=:none, outputdim::Int64)

    # Validate that the activation function is appropriate for output layers
    if !in(activation, [:softmax, :logistic, :regression])
        error("Output layer activation must be one of :softmax, :logistic, :regression. Input was :$activation")
    end

    # Validate that output is positive
    if outputdim <= 0
        error("Output layer must have at least 1 output unit. Input was $outputdim")
    end

    # Create LayerSpec - output layers are essentially linear layers with specific activations
    # normalization defaults to :none and should not be specified for output layers
    LayerSpec(name=name, kind=:linear, activation=activation, isoutput=true,
        optimization=optimization, outputdim=outputdim)
end

# ============================
# Structs for layers: hold pre-allocated weights, bias, data storage
# ============================

# Base.@kwdef mutable struct Slice
#     # for conv, some shared with maxpool
#     z4::SubArray{ELT,4} = ELT[;;;;]
#     znorm4::SubArray{ELT,4} = ELT[;;;;]
#     a4::SubArray{ELT,4} = ELT[;;;;]
#     a_below4::SubArray{ELT,4} = ELT[;;;;]
#     pad_a_below::SubArray{ELT,4} = ELT[;;;;]
#     eps_l4::SubArray{ELT,4} = ELT[;;;;]
#     pad_above_eps::SubArray{ELT,4} = ELT[;;;;]
#     grad_a4::SubArray{ELT,4} = ELT[;;;;]
#     pad_x::SubArray{ELT,4} = ELT[;;;;]

#     # for linear, some shared with flatten
#     z::SubArray{ELT,2} = ELT[;;]       # feed forward linear combination result
#     z_norm::SubArray{ELT,2} = ELT[;;]  # if doing batchnorm
#     a::SubArray{ELT,2} = ELT[;;]       # feed forward activation output
#     grad_a::SubArray{ELT,2} = ELT[;;]  # backprop derivative of activation output
#     a_below::SubArray{ELT,2} = ELT[;;]  
#     eps_l::SubArray{ELT,2} = ELT[;;]   # backprop error of the layer

#     # for maxpooling
#     # a: use a4
#     mask::SubArray{Bool,4} = Bool[;;;;]
#     # eps_l: use eps_l4

#     # for flatten
#     dl_dflat::SubArray{ELT,2} = ELT[;;]
#     # a: use a
#     # eps_l: use eps_l4

# end


Base.@kwdef struct ConvLayer <: Layer
    # data arrays
    z::Array{ELT,4}  
    z_norm::Array{ELT,4}    # if doing batchnorm: 2d to simplify batchnorm calcs
    a::Array{ELT,4}  
    a_below::Array{ELT,4}  
    pad_a_below::Array{ELT,4}  
    eps_l::Array{ELT,4}  
    pad_above_eps::Array{ELT,4}    # TODO need to test if this is needed given successive conv layer sizes
    grad_a::Array{ELT,4}  
    pad_x::Array{ELT,4}  

    # slices for minibatches of differing lengths
    # slices::Slice

    # special minibatch range used for data arrays
    mb_rng::Ref{UnitRange{Int}}

    # weight arrays
    weight::Array{ELT,4}  # = ELT[;;;;]  # (filter_h, filter_w, in_channels, out_channels)
    bias::Vector{ELT}  # = ELT[]    # (out_channels)
    grad_weight::Array{ELT,4}  # = ELT[;;;;]
    grad_bias::Vector{ELT}  # = ELT[]

    # cache arrays for optimization (only initialize and allocate if using)
    grad_m_weight::Array{ELT,4}
    grad_m_bias::Vector{ELT}
    grad_v_weight::Array{ELT,4}
    grad_v_bias::Vector{ELT}

    # layer specific functions: DO NOT USE DEFAULTS. defaults force the type and later assignment won't change it
    activationf::Function
    activation_gradf::Function
    normalizationf::Function
    normalization_gradf::Function

    # structs of layer specific parameters
    normparams::NormParam     # initialize to noop that won't allocate
    optparams::OptParam

    # scalar parameters
    name::Symbol
    optimization::Symbol
    adj::ELT
    padrule::Symbol  # can be :same or :none
    pad::Int64
    stride::Int64
    dobias::Bool
    # doslice::Bool
    isoutput::Bool
end

# helper functions for conv layers
same_pad(imgx,filx,stride) = div((imgx * (stride - 1) - stride + filx), 2)
dim_out(imgx, filx, stride, pad) = div(imgx - filx + 2pad, stride) + 1


# this method assigns every field with default initialization or values based on layerspec inputs
function ConvLayer(lr::LayerSpec, prevlayer, n_samples)
    outch = lr.outch
    prev_h, prev_w, inch, _ = size(prevlayer.a)

    # slice = Slice()

    # TODO: this assumes square images and square filters!  FIX
    pad = ifelse(lr.padrule == :same, same_pad(prev_h,lr.f_h, lr.stride), 0)  
    # output image dims: calculated once rather than over and over in training loop
    out_h = dim_out(prev_h, lr.f_h, lr.stride, pad)
    out_w = dim_out(prev_w, lr.f_w, lr.stride, pad)

    if lr.normalization == :batchnorm
        normalizationf = batchnorm!
        normalization_gradf = batchnorm_grad!
        normparams = BatchNorm{Vector{ELT}}(gam=ones(ELT, outch), bet=zeros(ELT, outch),
            grad_gam=zeros(ELT, outch), grad_bet=zeros(ELT, outch),
            grad_m_gam=zeros(ELT, outch), grad_v_gam=zeros(ELT, outch),
            grad_m_bet=zeros(ELT, outch), grad_v_bet=zeros(ELT, outch),
            mu=zeros(ELT, outch), stddev=zeros(ELT, outch),
            mu_run=zeros(ELT, outch), std_run=zeros(ELT, outch))
        dobias = false
    elseif lr.normalization == :none
        normalizationf = noop
        normalization_gradf = noop
        normparams = NoNorm() # initialize as empty struct of different type
        dobias = true
    else
        error("Only :batchnorm and :none  supported, not $(Symbol(lr.normalization)).")
    end

    if lr.activation == :relu
        activationf = relu!
    elseif lr.activation == :leaky_relu
        activationf = leaky_relu!
    elseif lr.activation == :none
        activationf = noop
    else
        error("Only :relu, :leaky_relu and :none  supported, not $(Symbol(lr.activation)).")
    end

    if lr.activation == :relu
        activation_gradf = relu_grad!
    elseif lr.activation == :leaky_relu
        activation_gradf = leaky_relu_grad!
    elseif lr.activation == :none
        activation_gradf = noop
    else
        error("Only :relu, :leaky_relu and :none  supported, not $(Symbol(lr.activation)).")
    end

    if (lr.optimization == :adam) | (lr.optimization == :adamw)
        optparams = AdamParam(b1=0.9, b2=0.999, decay=0.01)
    elseif lr.optimization == :none
        optparams = NoOpt()
    else
        error("Only :none, :adam or :adamw supported, not $(Symbol(lr.optimization)).")
    end

    ConvLayer(
        # data arrays
        pad_x=zeros(ELT, out_h + 2pad, out_w + 2pad, inch, n_samples),  # TODO this isn't right for padrule=:none
        a_below=zeros(ELT, prev_h, prev_w, inch, n_samples),
        pad_a_below=zeros(ELT, prev_h + 2pad, prev_w + 2pad, inch, n_samples),
        z=zeros(ELT, out_h, out_w, outch, n_samples),
        z_norm=ifelse(lr.normalization == :batchnorm, zeros(ELT, out_h, out_w, outch, n_samples), zeros(ELT, 0, 0, 0, 0)),
        a=zeros(ELT, out_h, out_w, outch, n_samples),
        eps_l=zeros(ELT, prev_h, prev_w, inch, n_samples),
        grad_a=zeros(ELT, out_h, out_w, outch, n_samples),
        pad_above_eps=zeros(ELT, prev_h, prev_w, outch, n_samples),
        mb_rng = Ref(1:n_samples),

        # slice = Slice(z4=view(z,:,:,:,:), znorm4=view(z_norm,:,:,:,:), a4=view(a,:,:,:,:),
        #             a_below4=view(a_below,:,:,:,:), pad_a_below=view(pad_a_below,:,:,:,:),
        #             eps_l4=view(eps_l,:,:,:,:), pad_above_eps=view(pad_a_above,:,:,:,:),
        #             grad_a4=view(grad_a,:,:,:,:), pad_x=view(pad_x,:,:,:,:)),
        # slice = Slice(),

        # weight arrays
        weight=he_initialize((lr.f_h, lr.f_w, inch, outch), scale=2.2, adj=lr.adj),
        bias=zeros(ELT, outch),
        grad_weight=zeros(ELT, lr.f_h, lr.f_w, inch, outch),
        grad_bias=zeros(ELT, outch),

        # cache arrays for optimization (only initialize and allocate if using)
        grad_m_weight=ifelse(isa(optparams, AdamParam), zeros(ELT, lr.f_h, lr.f_w, inch, lr.outch), zeros(ELT, 0, 0, 0, 0)),
        grad_m_bias=ifelse(isa(optparams, AdamParam), zeros(ELT, outch), zeros(ELT, 0)),
        grad_v_weight=ifelse(isa(optparams, AdamParam), zeros(ELT, lr.f_h, lr.f_w, inch, lr.outch), zeros(ELT, 0, 0, 0, 0)),
        grad_v_bias=ifelse(isa(optparams, AdamParam), zeros(ELT, outch), zeros(ELT, 0)),

        # structs of layer specific parameters
        normparams=normparams,
        optparams=optparams,

        # layer specific functions
        activationf=activationf,
        activation_gradf=activation_gradf,
        normalizationf=normalizationf,
        normalization_gradf=normalization_gradf,

        # scalar parameters
        name=lr.name,
        adj=lr.adj,
        optimization=lr.optimization,
        padrule=lr.padrule,
        pad=pad,
        stride=lr.stride,
        dobias=dobias,
        # doslice=false,
        isoutput=lr.isoutput
    )
end



Base.@kwdef struct LinearLayer <: Layer
    # data array views
    z::Array{ELT,2}  #     = ELT[;;]       # feed forward linear combination result
    z_norm::Array{ELT,2} #      = ELT[;;]  # if doing batchnorm
    a::Array{ELT,2}  #     = ELT[;;]      # feed forward activation output
    grad_a::Array{ELT,2}  #   = ELT[;;]  # backprop derivative of activation output
    a_below::Array{ELT,2}  #   = ELT[;;]
    eps_l::Array{ELT,2}   #   = ELT[;;]   # backprop error of the layer

    # slices for minibatches of differing lengths
    # slices::Slice

    # special minibatch range used for data arrays
    mb_rng::Ref{UnitRange{Int}}

    # weight arrays
    weight::Array{ELT,2}     # = ELT[;;] # (output_dim, input_dim)
    bias::Vector{ELT}        # = ELT[]     # (output_dim)
    grad_weight::Array{ELT,2}   #     = ELT[;;]
    grad_bias::Vector{ELT}  #   = ELT[]

    # cache arrays for optimization (only initialize and allocate if using)
    grad_m_weight::Array{ELT,2}
    grad_m_bias::Vector{ELT}
    grad_v_weight::Array{ELT,2}
    grad_v_bias::Vector{ELT}

    # structs of layer specific parameters
    normparams::NormParam  #       = NoNorm()
    optparams::OptParam

    # layer specific functions: DO NOT USE DEFAULTS. defaults force the type and later assignment won't change it
    activationf::Function
    activation_gradf::Function
    normalizationf::Function
    normalization_gradf::Function

    # scalar parameters
    name::Symbol
    optimization::Symbol
    adj::ELT
    dobias::Bool
    # doslice::Bool
    isoutput::Bool   # is this the output layer?
end

# this method assigns every field with default initialization or values based on layerspec inputs
function LinearLayer(lr::LayerSpec, prevlayer, n_samples)
    outputdim = lr.outputdim      # rows
    inputs = size(prevlayer.a, 1)    # rows of lower layer output become columns

    if lr.normalization == :batchnorm
        normalizationf = batchnorm!
        normalization_gradf = batchnorm_grad!
        normparams = BatchNorm{Vector{ELT}}(gam=ones(ELT, outputdim), bet=zeros(ELT, outputdim),
            grad_gam=zeros(ELT, outputdim), grad_bet=zeros(ELT, outputdim),
            grad_m_gam=zeros(ELT, outputdim), grad_v_gam=zeros(ELT, outputdim),
            grad_m_bet=zeros(ELT, outputdim), grad_v_bet=zeros(ELT, outputdim),
            mu=zeros(ELT, outputdim), stddev=zeros(ELT, outputdim),
            mu_run=zeros(ELT, outputdim), std_run=zeros(ELT, outputdim))
        dobias = false
    elseif lr.normalization == :none
        normalizationf = noop
        normalization_gradf = noop
        normparams = NoNorm()  # initialize as empty struct of different type
        dobias = true
    else
        error("Only :batchnorm and :none  supported, not $(Symbol(lr.normalization)).")
    end

    if lr.activation == :relu
        activationf = relu!
    elseif lr.activation == :leaky_relu
        activationf = leaky_relu!
    elseif lr.activation == :none
        activationf = noop
    elseif lr.activation == :softmax
        activationf = softmax!
    elseif lr.activation == :logistic   # rarely used any more
        activationf = logistic!
    elseif lr.activation == :regression
        activationf = regression!
    else
        error("Only :relu, :leaky_relu, :softmax and :none  supported, not $(Symbol(lr.activation)).")
    end

    if lr.activation == :relu  # this has no effect on the output layer, but need it for hidden layers
        activation_gradf = relu_grad!
    elseif lr.activation == :leaky_relu
        activation_gradf = leaky_relu_grad!
    elseif lr.activation == :softmax
        activation_gradf = noop
    elseif lr.activation == :none
        activation_gradf = noop
    elseif lr.activation == :regression
        activation_gradf = noop
    else
        error("Only :relu, :leaky_relu, :softmax, :regression and :none  supported, not $(Symbol(lr.activation)).")
    end

    if (lr.optimization == :adam) | (lr.optimization == :adamw)
        optparams = AdamParam(b1=0.9, b2=0.999, decay=0.01)
        optimization = lr.optimization
    elseif lr.optimization == :none
        optparams = NoOpt()
    else
        error("Only :none, :adam or :adamw supported, not $(Symbol(lr.optimization)).")
    end

    LinearLayer(
        # data arrays
        z=zeros(ELT, outputdim, n_samples),
        z_norm=ifelse(lr.normalization == :batchnorm, zeros(ELT, outputdim, n_samples), zeros(ELT, 0, 0)),
        a=zeros(ELT, outputdim, n_samples),
        a_below=zeros(ELT, inputs, n_samples),
        eps_l=zeros(ELT, outputdim, n_samples),
        grad_a=zeros(ELT, outputdim, n_samples),

        mb_rng = Ref(1:n_samples),
        # slice = Slice(z=view(z,:,:), znorm=view(z_norm,:,:), a=view(a,:,:),
        #             a_below=view(a_below,:,:), eps_l=view(eps_l,:,:), grad_a=view(grad_a,:,:)),
        # slice = Slice(),

        # weight arrays
        weight=he_initialize((outputdim, inputs), scale=1.5, adj=lr.adj),
        bias=zeros(ELT, outputdim),
        grad_weight=zeros(outputdim, inputs),
        grad_bias=zeros(ELT, outputdim),

        # cache arrays for optimization (only initialize and allocate if using)
        grad_m_weight=ifelse(isa(optparams, AdamParam), zeros(outputdim, inputs), zeros(ELT, 0, 0)),
        grad_m_bias=ifelse(isa(optparams, AdamParam), zeros(ELT, outputdim), zeros(ELT, 0)),
        grad_v_weight=ifelse(isa(optparams, AdamParam), zeros(ELT, outputdim, inputs), zeros(ELT, 0, 0)),
        grad_v_bias=ifelse(isa(optparams, AdamParam), zeros(ELT, outputdim), zeros(ELT, 0)),

        # structs of layer specific parameters
        normparams=normparams,
        optparams=optparams,

        # layer specific functions
        activationf=activationf,
        activation_gradf=activation_gradf,
        normalizationf=normalizationf,
        normalization_gradf=normalization_gradf,

        # scalar parameters
        name=lr.name,
        optimization=lr.optimization,
        adj=lr.adj,
        dobias=dobias,
        # doslice=false,
        isoutput=lr.isoutput    
    )
end


# no weight, bias, gradients, activation
Base.@kwdef struct FlattenLayer <: Layer
    name::Symbol = :noname
    outputdim::Int64   # must supply input value in constructor
    dl_dflat::Array{ELT,2} = ELT[;;]
    a::Array{ELT,2} = ELT[;;]
    eps_l::Array{ELT,4} = ELT[;;;;]
    isoutput::Bool
    # doslice::Bool

    # slices for minibatches of differing lengths
    # slices::Slice

    # special minibatch range used for data arrays
    mb_rng::Ref{UnitRange{Int}}
end

# constructor method to prepare inputs and create layer
function FlattenLayer(lr::LayerSpec, prevlayer, n_samples)
    h, w, ch, _ = size(prevlayer.a)
    outputdim = h * w * ch

    FlattenLayer(
        name=lr.name,
        outputdim=outputdim,
        a=zeros(ELT, outputdim, n_samples),
        dl_dflat=zeros(ELT, outputdim, n_samples),
        eps_l=zeros(ELT, h, w, ch, n_samples),
        isoutput=lr.isoutput,
        mb_rng = Ref(1:n_samples),
        # doslice=false,

        # slice = Slice(dl_dlfat=view(dl_dflat,:,:), a=view(a,:,:), eps_l=view(eps_l,:,:))
        # slice = Slice()
    )
end



Base.@kwdef struct InputLayer <: Layer     # we only have this to simplify feedforward loop
    name::Symbol = :noname
    kind::Symbol   # must supply input value in constructor
    out_h::Int64 = 0
    out_w::Int64 = 0
    outch::Int64 = 0
    outputdim::Int64 = 0
    a::Array{ELT}   # no default provided because dims different for :image vs :linear

    # special minibatch range used for data arrays
    mb_rng::Ref{UnitRange{Int}}
end

function InputLayer(lr::LayerSpec, n_samples)
    if lr.outputdim > 0  # dense input layer
        InputLayer(name=lr.name, kind=:linear,
            outputdim=lr.outputdim, mb_rng = Ref(1:n_samples),
            a=zeros(ELT, lr.outputdim, n_samples))
    elseif lr.outch > 0  # image input layer
        InputLayer(name=lr.name, kind=:image,
            out_h=lr.h, out_w=lr.w, outch=lr.outch, mb_rng = Ref(1:n_samples),
            a=zeros(ELT, lr.h, lr.w,  lr.outch, n_samples))
    end
end

Base.@kwdef struct MaxPoolLayer <: Layer
    name::Symbol = :noname
    pool_size::Tuple{Int,Int}
    a::Array{ELT,4} = ELT[;;;;]
    mask::Array{Bool,4} = Bool[;;;;]
    eps_l::Array{ELT,4} = ELT[;;;;]
    isoutput::Bool
    # doslice::Bool

    # slices for minibatches of differing lengths
    # slices::Slice

    # special minibatch range used for data arrays
    mb_rng::Ref{UnitRange{Int}}
end

# constructor method to prepare inputs and create layer
function MaxPoolLayer(lr::LayerSpec, prevlayer, n_samples)

    in_h, in_w, outch, _ = size(prevlayer.grad_a)
    out_h = div(in_h, lr.f_h) # assume stride = lr.f_h implicit in code
    out_w = div(in_w, lr.f_w)  # ditto
    batch_size = n_samples

    MaxPoolLayer(
        name=lr.name,
        pool_size=(lr.f_h, lr.f_w),
        a=zeros(ELT, out_h, out_w, outch, batch_size),
        mask=falses(in_h, in_w, outch, batch_size),
        eps_l=zeros(ELT, in_h, in_w, outch, batch_size),
        isoutput=lr.isoutput,
        # doslice=false,
        mb_rng = Ref(1:batch_size),

        # slice=Slice()
        # slice = Slice(mask=view(mask,:,:,:,:), a=view(a,:,:,:,:), eps_l=view(eps_l,:,:,:,:))
    )
end
