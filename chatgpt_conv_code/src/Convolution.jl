"""
Module Convolution:

"""
module Convolution


# ----------------------------------------------------------------------------------------

include("sample_code.jl")

# data structures for neural network
export 
    LayerSpec,
    ConvLayer, 
    LinearLayer, 
    FlattenLayer, 
    MaxPoolLayer


# functions you can use
export 
    preptrain,
    allocate_layers, 
    setup_preds,
    train_loop!,
    predict,
    display_mnist_digit,
    weights2file,
    file2weights

using Random
using LinearAlgebra
using Colors, Plots
using Serialization
using MLDatasets
using StatsBase
using Statistics
using LinearAlgebra

end  # module ConvLayer