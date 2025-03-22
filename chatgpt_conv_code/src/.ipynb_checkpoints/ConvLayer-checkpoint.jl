"""
Module ConvLayer:

Includes the following functions to run directly:

- train() -- train neural networks for up to 9 hidden layers
- setup_params -- load hyper_parameters and create Hyper_parameters object
- extract_data() -- extracts data for MNIST from matlab files
- shuffle_data!() -- shuffles training examples (do this before minibatch training)
- test_score() -- cost and accuracy given test data and saved theta
- save_params() -- save all model parameters
- load_params() -- load all model parameters
- predictions_vector() -- predictions given x and saved theta
- accuracy() -- calculates accuracy of predictions compared to actual labels
- normalize_inputs() -- normalize via standardization (mean, sd) or minmax
- normalize_replay!() --
- nnpredict() --  calculate predictions from previously trained parameters and new data
- display_mnist_digit() --  show a digit
- wrong_preds() -- return the indices of wrong predictions against supplied correct labels
- right_preds() -- return the indices of correct predictions against supplied correct labels
- plot_output() -- plot learning, cost for training and test for previously saved training runs

Enter using .ConvLayer to use the module.

These data structures are used to hold parameters and data:

- Wgts holds theta, bias, delta_th, delta_b, theta_dims, output_layer, k
- Model_data holds inputs, targets, a, z, z_norm, epsilon, gradient_function
- Batch_norm_params holds gam (gamma), bet (beta) for batch normalization and intermediate
    data used for backprop with batch normalization: delta_gam, delta_bet, 
    delta_z_norm, delta_z, mu, stddev, mu_run, std_run
- Hyper_parameters holds user input hyper_parameters and some derived parameters.
- Batch_view holds views on model data to hold the subset of data used in minibatches. 
    (This is not exported as there is no way to use it outside of the backprop loop.)
- Model_def holds the string names and functions that define the layers of the training
    model.

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