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
    predict,
    train_loop!,
    predict,
    display_mnist_digit,
    weights2file,
    file2weights,
    show_all_array_sizes


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