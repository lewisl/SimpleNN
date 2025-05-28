"""
Module Convolution:

"""
module SimpleNN


# ----------------------------------------------------------------------------------------

const ELT = Float32   # ELT for element type, Float32 or Float64
const IT = 1f-12   # IT for itty bitty

using Random
using LinearAlgebra
using Colors, Plots
using Serialization
using MLDatasets
using StatsBase
using Statistics
using BenchmarkTools

include("data_layers.jl")
include("layer_functions.jl")
include("modifiers.jl")
include("training.jl")
include("mnist_fun.jl")

# data structures for neural network
export
    ConvLayer,
    LinearLayer,
    FlattenLayer,
    MaxPoolLayer,
    HyperParameters

# creating layerspecs to define a model
export
    LayerSpec,          # constructor for a specification of any type of layer
    convlayerspec,      # constructor with only inputs needed for a convolutional layer
    linearlayerspec,    # constructor with only inputs needed for a linear layer
    flattenlayerspec,   # constructor with only inputs needed for a flatten layer
    maxpoollayerspec,    # constructor with only inputs needed for a maxpooling layer
    inputlayerspec,      # constructor for the spec of the input layer
    outputlayerspec     # constructor for an output layer

# functions you can use
export
    setup_train,
    setup_mnist,
    allocate_layers,
    setup_preds,
    prediction,
    minibatch_prediction,
    train_loop!,
    prediction,
    display_mnist_digit,
    weights2file,
    file2weights,
    show_all_array_sizes,
    plot_stats,
    feedforward!,
    backprop!,
    update_weight_loop!,
    update_weights!,
    update_batchnorm!,
    pre_adam!,
    pre_adam_batchnorm!,
    adam_helper!



end  # module ConvLayer
