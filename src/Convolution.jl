"""
Module Convolution:

"""
module Convolution


# ----------------------------------------------------------------------------------------

include("layer_functions.jl")
include("training.jl")
include("mnist_fun.jl")

# data structures for neural network
export
    LayerSpec,
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
    maxpoollayerspec    # constructor with only inputs needed for a maxpooling layer

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
    plot_stats

using Random
using LinearAlgebra
using Colors, Plots
using Serialization
using MLDatasets
using StatsBase
using Statistics
using LinearAlgebra
using BenchmarkTools

end  # module ConvLayer
