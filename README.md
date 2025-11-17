# SimpleNN.jl

A simple neural network library implemented in Julia, designed for educational and research purposes. This project provides the basic building blocks for creating, training, and evaluating neural networks from scratch.

## Features

- **Core Layers:** Implementation of fundamental neural network layers like Dense and Convolutional layers.
- **Training Infrastructure:** Includes a training loop, optimizers, and loss functions with manually implemented backpropagation for gradient computation.
- **In-depth Regression Analysis:** Provides tools for regression tasks using gradient descent, including common analysis of variance (ANOVA) metrics to evaluate model fit.
- **Data Handling:** Utilities for loading and preprocessing datasets, with support for `MLDatasets.jl`.
- **Examples:** Comes with ready-to-run examples for tasks like:
    - MNIST digit classification (`src/mnist_fun.jl`)
    - Regression analysis (`src/runregression.jl`, `src/cars_analysis.jl`)
    - Convolutional network demonstrations (`src/runconv.jl`)

## Project Structure

```text
.
├── Project.toml      # Julia project dependencies
├── README.md         # This file
├── src/              # Main source code
│   ├── SimpleNN.jl        # Main module file, loads all sub-modules
│   ├── cars_analysis.jl   # Example script for regression on the 'mtcars' dataset
│   ├── data_layers.jl     # Utilities for loading datasets and managing data batches
│   ├── layer_functions.jl # Core neural network layers (Dense, Conv) and their logic
│   ├── mnist_fun.jl       # Functions to build and train a model on the MNIST dataset
│   ├── modifiers.jl       # Activation functions (ReLU, Softmax) and their derivatives
│   ├── profile_runconv.jl # Script for performance profiling of convolutional layers
│   ├── regr_fun.jl        # Functions for regression using gradient descent and ANOVA
│   ├── runconv.jl         # Example script to run a convolutional neural network
│   ├── runregression.jl   # Example script to execute a regression analysis
│   ├── training.jl        # Main training loop, optimizers, and loss functions
│   ├── utility.jl         # General helper and utility functions
│   └── ...
├── test/             # Test suite and run scripts
│   ├── test.jl       # Main test file
│   └── ...           # Specific tests and runners
└── docs/             # Documentation and analysis notes
```

## Dependencies

The project relies on the following Julia packages as defined in `Project.toml`:

- `BenchmarkTools`
- `CSV`
- `Colors`
- `Distributions`
- `GLM`
- `LazyTables`
- `LinearAlgebra`
- `LoopVectorization`
- `MLDatasets`
- `Plots`
- `Random`
- `Serialization`
- `Statistics`
- `StatsBase`

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd SimpleNN.jl
    ```

2.  **Instantiate the Julia environment:**
    Open a Julia REPL in the project directory and run:
    ```julia
    using Pkg
    Pkg.activate(".")
    Pkg.instantiate()
    ```

## Usage

You can run the examples and tests from the `src` and `test` directories.

### Example: Run MNIST Classification

To train a model on the MNIST dataset, you can include and run the functions from `src/mnist_fun.jl`.

```julia
# In a Julia REPL from the project root
include("src/mnist_fun.jl")

# This file contains functions to build, train, and test a model.
# You can call them as needed after inspecting the file for exact function names.
```

### Example: Run a Regression Task

Similarly, to run the regression analysis:

```julia
# In a Julia REPL from the project root
include("src/runregression.jl")

# This will execute the regression analysis defined in the file.
```

This project is a great starting point for anyone looking to understand the inner workings of neural networks by exploring a hands-on implementation in Julia.
