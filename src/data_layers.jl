

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
    name::Symbol = :noname
    kind::Symbol = :none
    activation::Symbol = :none
    normalization::Symbol = :none  # options are :none, :batchnorm
    optimization::Symbol = :none  # options are :none :adam :adamw
    adj::Float64 = 0              # leaky_relu factor. also for he_initialize
    h::Int64 = 0                  # image height (rows) or output neurons for linear layers
    w::Int64 = 0                  # image width (columns)
    outch::Int64 = 0              # output channels in image format array
    f_h::Int64 = 0                # filter height (rows)
    f_w::Int64 = 0                # filter width (columns)
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
function convlayerspec(; name::Symbol, activation::Symbol=:relu, normalization::Symbol=:none, optimization::Symbol=:none, 
        adj::Float64=0.002, h::Int64=0, w::Int64=0, outch::Int64, f_h::Int64, f_w::Int64, inch::Int64=0, padrule::Symbol=:same)
    LayerSpec(name=name, kind=:conv, activation=activation, normalization=normalization, optimization=optimization, 
                adj=adj, h=h, w=w, outch=outch, f_h=f_h, f_w=f_w, inch=inch, padrule=padrule)
end

function linearlayerspec(; name::Symbol, activation::Symbol=:relu, normalization::Symbol=:none, optimization::Symbol=:none, 
        adj::Float64=0.002, output::Int64)
    LayerSpec(name=name, kind=:linear, activation=activation, normalization=normalization, optimization=optimization, adj=adj, h=output)
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

    inputlayerspec(name=:input_array, output=15)
Or supply only an input for the output parameter for a dense linear input array of dimensions output, batchsize.

For both array inputs, the batchsize will be determined from the actual array provided as x input to function trainloop!.
"""
function inputlayerspec(; name::Symbol, h::Int64=0, w::Int64=0, outch::Int64=0, output::Int64=0)
    if (h>0) & (w>0) & (outch>0) 
        LayerSpec(name=name, kind=:input, h=h, w=w, outch=outch)
    elseif output > 0
        LayerSpec(name=name, kind=:input, h=output)
    else
        error("Positive integer inputs must be supplied for h, w, AND outch OR only for output.")
    end
end



# ============================
# Structs for layers: hold pre-allocated weights, bias, data storage
# ============================


Base.@kwdef struct ConvLayer <: Layer  
    # data arrays
    z::Array{Float64,4}  # = Float64[;;;;]
    z_norm::Array{Float64,4}  # = Float64[;;;;]  # if doing batchnorm: 2d to simplify batchnorm calcs
    a::Array{Float64,4}  # = Float64[;;;;]
    a_below::Array{Float64,4}  # = Float64[;;;;]
    pad_a_below::Array{Float64,4}  # = Float64[;;;;]
    eps_l::Array{Float64,4}  # = Float64[;;;;]
    pad_next_eps::Array{Float64,4}  # = Float64[;;;;]  # TODO need to test if this is needed given successive conv layer sizes
    grad_a::Array{Float64,4}  # = Float64[;;;;]
    pad_x::Array{Float64,4}  # = Float64[;;;;]

    # weight arrays
    weight::Array{Float64,4}  # = Float64[;;;;]  # (filter_h, filter_w, in_channels, out_channels)
    bias::Vector{Float64}  # = Float64[]    # (out_channels)
    grad_weight::Array{Float64,4}  # = Float64[;;;;]
    grad_bias::Vector{Float64}  # = Float64[]

    # cache arrays for optimization (only initialize and allocate if using)
    grad_m_weight::Array{Float64, 4}
    grad_m_bias::Vector{Float64}
    grad_v_weight::Array{Float64,4}
    grad_v_bias::Vector{Float64}

    # layer specific functions: DO NOT USE DEFAULTS. defaults force the type and later assignment won't change it
    activationf::Function
    activation_gradf::Function
    normalizationf::Function
    normalization_gradf::Function

    # structs of layer specific parameters
    normparams::NormParam     # initialize to noop that won't allocate
    optparams::OptParam

    # scalar parameters
    name::Symbol  # = :noname
    optimization::Symbol  # = :none
    adj::Float64  # = 0.0
    padrule::Symbol  # = :same   # other option is :none
    stride::Int64  # = 1     # assume stride is symmetrical for now
    dobias::Bool  # = true
end

# this method assigns every field with default initialization or values based on layerspec inputs
function ConvLayer(lr::LayerSpec, prevlayer, n_samples)
    outch = lr.outch
    prev_h, prev_w, inch, _ = size(prevlayer.a)

    pad = ifelse(lr.padrule == :same, 1, 0)
    # output image dims: calculated once rather than over and over in training loop
    out_h = div((prev_h + 2pad - lr.f_h), lr.stride) + 1
    out_w = div((prev_w + 2pad - lr.f_w), lr.stride) + 1
    if lr.normalization == :batchnorm
            normalizationf = batchnorm!
            normalization_gradf = batchnorm_grad!
            normparams=BatchNorm{Vector{Float64}}(gam=ones(outch), bet=zeros(outch),
                grad_gam=zeros(outch), grad_bet=zeros(outch),
                grad_m_gam=zeros(outch), grad_v_gam=zeros(outch),
                grad_m_bet=zeros(outch), grad_v_bet=zeros(outch),
                mu=zeros(outch), stddev=zeros(outch),
                mu_run=zeros(outch), std_run=zeros(outch))
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
            activationf=relu!
        elseif lr.activation == :leaky_relu
            activationf=leaky_relu!
        elseif lr.activation == :none
            activationf=noop
        else
            error("Only :relu, :leaky_relu and :none  supported, not $(Symbol(lr.activation)).")
        end
        if lr.activation == :relu
            activation_gradf=relu_grad!
        elseif lr.activation == :leaky_relu
            activation_gradf=leaky_relu_grad!
        elseif lr.activation == :none
            activation_gradf=noop
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
        pad_x=zeros(out_h + 2pad, out_w + 2pad, inch, n_samples),
        a_below = zeros(prev_h, prev_w, inch, n_samples),
        pad_a_below=zeros(prev_h + 2pad, prev_w + 2pad, inch, n_samples),
        z=zeros(out_h, out_w, outch, n_samples),
        z_norm=ifelse(lr.normalization == :batchnorm, zeros(out_h, out_w, outch, n_samples), zeros(0, 0, 0, 0)),
        a=zeros(out_h, out_w, outch, n_samples),
        eps_l=zeros(prev_h, prev_w, inch, n_samples),
        grad_a=zeros(out_h, out_w, outch, n_samples),
        pad_next_eps=zeros(prev_h, prev_w, outch, n_samples),

        # weight arrays
        weight=he_initialize((lr.f_h, lr.f_w, inch, lr.outch), scale=2.2, adj=lr.adj),
        bias=zeros(outch),
        grad_weight=zeros(lr.f_h, lr.f_w, inch, outch),
        grad_bias=zeros(outch),

        # cache arrays for optimization (only initialize and allocate if using)
        grad_m_weight = ifelse(isa(optparams,AdamParam),zeros(lr.f_h, lr.f_w, inch, lr.outch),zeros(0,0,0,0)),
        grad_m_bias = ifelse(isa(optparams,AdamParam),zeros(outch),zeros(0)),
        grad_v_weight = ifelse(isa(optparams,AdamParam),zeros(lr.f_h, lr.f_w, inch, lr.outch),zeros(0,0,0,0)),
        grad_v_bias = ifelse(isa(optparams,AdamParam),zeros(outch),zeros(0)),

        # structs of layer specific parameters
        normparams = normparams,
        optparams = optparams,

        # layer specific functions
        activationf = activationf,
        activation_gradf = activation_gradf,
        normalizationf=normalizationf,
        normalization_gradf=normalization_gradf,

        # scalar parameters
        name=lr.name,
        adj=lr.adj,
        optimization=lr.optimization,
        padrule=lr.padrule,
        stride=lr.stride,
        dobias=dobias
    )
end


Base.@kwdef struct LinearLayer <: Layer
    # data arrays
    z::Array{Float64,2}  #     = Float64[;;]       # feed forward linear combination result
    z_norm::Array{Float64,2} #      = Float64[;;]  # if doing batchnorm
    a::Array{Float64,2}  #     = Float64[;;]      # feed forward activation output
    grad_a::Array{Float64,2}  #   = Float64[;;]  # backprop derivative of activation output
    a_below::Array{Float64,2}  #   = Float64[;;]
    eps_l::Array{Float64,2}   #   = Float64[;;]   # backprop error of the layer

    # weight arrays
    weight::Array{Float64,2}     # = Float64[;;] # (output_dim, input_dim)
    bias::Vector{Float64}        # = Float64[]     # (output_dim)
    grad_weight::Array{Float64,2}   #     = Float64[;;]
    grad_bias::Vector{Float64}  #   = Float64[]

    # cache arrays for optimization (only initialize and allocate if using)
    grad_m_weight::Array{Float64, 2}
    grad_m_bias::Vector{Float64}
    grad_v_weight::Array{Float64,2}
    grad_v_bias::Vector{Float64}

    # structs of layer specific parameters
    normparams::NormParam  #       = NoNorm()
    optparams::OptParam

    # layer specific functions: DO NOT USE DEFAULTS. defaults force the type and later assignment won't change it
    activationf::Function
    activation_gradf::Function
    normalizationf::Function
    normalization_gradf::Function

    # scalar parameters
    name::Symbol       #  = :noname
    optimization::Symbol     # = :none
    adj::Float64   #      = 0.0
    dobias::Bool    #             = true
end

# this method assigns every field with default initialization or values based on layerspec inputs
function LinearLayer(lr::LayerSpec, prevlayer, n_samples)
    outputs = lr.h        # rows
    inputs = size(prevlayer.a, 1)    # rows of lower layer output become columns
    if lr.normalization == :batchnorm
            normalizationf = batchnorm!
            normalization_gradf = batchnorm_grad!
            normparams=BatchNorm{Vector{Float64}}(gam=ones(outputs), bet=zeros(outputs),
                grad_gam=zeros(outputs), grad_bet=zeros(outputs),
                grad_m_gam=zeros(outputs), grad_v_gam=zeros(outputs),
                grad_m_bet=zeros(outputs), grad_v_bet=zeros(outputs),
                mu=zeros(outputs), stddev=zeros(outputs),
                mu_run=zeros(outputs), std_run=zeros(outputs))
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
            activationf=relu!
        elseif lr.activation == :leaky_relu
            activationf=leaky_relu!
        elseif lr.activation == :none
            activationf=noop
        elseif lr.activation == :softmax
            activationf=softmax!
        elseif lr.activation == :logistic   # rarely used any more
            activationf=logistic!
        elseif lr.activation == :regression
            activationf=regression!
        else
            error("Only :relu, :leaky_relu, :softmax and :none  supported, not $(Symbol(lr.activation)).")
        end
        if lr.activation == :relu  # this has no effect on the output layer, but need it for hidden layers
            activation_gradf=relu_grad!
        elseif lr.activation == :leaky_relu
            activation_gradf=leaky_relu_grad!
        elseif lr.activation == :softmax
            activation_gradf=noop
        elseif lr.activation == :none
            activation_gradf=noop
        else
            error("Only :relu, :leaky_relu, :softmax and :none  supported, not $(Symbol(lr.activation)).")
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
        z=zeros(outputs, n_samples),
        z_norm=ifelse(lr.normalization == :batchnorm, zeros(outputs, n_samples), zeros(0, 0)),
        a=zeros(outputs, n_samples),
        a_below = zeros(inputs, n_samples),
        eps_l=zeros(outputs, n_samples),
        grad_a=zeros(outputs, n_samples),

        # weight arrays
        weight=he_initialize((outputs, inputs), scale=1.5, adj=lr.adj),
        bias=zeros(outputs),
        grad_weight=zeros(outputs, inputs),
        grad_bias=zeros(outputs),

        # cache arrays for optimization (only initialize and allocate if using)
        grad_m_weight = ifelse(isa(optparams,AdamParam),zeros(outputs, inputs),zeros(0,0)),
        grad_m_bias = ifelse(isa(optparams,AdamParam),zeros(outputs),zeros(0)),
        grad_v_weight = ifelse(isa(optparams,AdamParam),zeros(outputs, inputs),zeros(0,0)),
        grad_v_bias = ifelse(isa(optparams,AdamParam),zeros(outputs),zeros(0)),

        # structs of layer specific parameters
        normparams = normparams,
        optparams = optparams,

        # layer specific functions
        activationf = activationf,
        activation_gradf = activation_gradf,
        normalizationf=normalizationf,
        normalization_gradf=normalization_gradf,

        # scalar parameters
        name=lr.name,
        optimization=lr.optimization,
        adj=lr.adj,
        dobias=dobias
        )
end

# no weight, bias, gradients, activation
Base.@kwdef struct FlattenLayer <: Layer
    name::Symbol = :noname
    output_dim::Int64 = 0
    dl_dflat::Array{Float64,2} = Float64[;;]
    a::Array{Float64,2} = Float64[;;]
    eps_l::Array{Float64,4} = Float64[;;;;]
end

# constructor method to prepare inputs and create layer
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

Base.@kwdef struct InputLayer <: Layer     # we only have this to simplify feedforward loop
    name::Symbol = :noname
    kind::Symbol = :image   # other allowed value is :linear
    out_h::Int64 = 0
    out_w::Int64 = 0
    outch::Int64 = 0
    a::Array{Float64}   # no default provided because dims different for :image vs :linear
end


Base.@kwdef struct MaxPoolLayer <: Layer
    name::Symbol = :noname
    pool_size::Tuple{Int,Int}
    a::Array{Float64,4} = Float64[;;;;]
    mask::Array{Bool,4} = Bool[;;;;]
    eps_l::Array{Float64,4} = Float64[;;;;]
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
        a=zeros(out_h, out_w, outch, batch_size),
        mask=falses(in_h, in_w, outch, batch_size),
        eps_l=zeros(in_h, in_w, outch, batch_size),
    )
end
