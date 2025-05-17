using Profile
using BenchmarkTools

# Include the SimpleNN module
include("SimpleNN.jl")
using .SimpleNN

# Create a small test network with ADAM optimization
function create_test_network()
    # Create a simple network with a few layers using ADAM
    layerspecs = [
        LayerSpec(name=:input, kind=:input, h=28, w=28, outch=1),
        LayerSpec(name=:conv1, kind=:conv, f_h=3, f_w=3, outch=32, stride=1, padrule=:same, activation=:relu, normalization=:batchnorm, optimization=:adam),
        LayerSpec(name=:pool1, kind=:maxpool, f_h=2, f_w=2, stride=2),
        LayerSpec(name=:conv2, kind=:conv, f_h=3, f_w=3, outch=64, stride=1, padrule=:same, activation=:relu, normalization=:batchnorm, optimization=:adam),
        LayerSpec(name=:pool2, kind=:maxpool, f_h=2, f_w=2, stride=2),
        LayerSpec(name=:flatten, kind=:flatten),
        LayerSpec(name=:linear1, kind=:linear, h=10, activation=:softmax, optimization=:adam)
    ]
    
    # Create a small batch of random data
    batch_size = 32
    x = rand(Float64, 28, 28, 1, batch_size)
    y = zeros(Float64, 10, batch_size)
    for i in 1:batch_size
        y[rand(1:10), i] = 1.0
    end
    
    return layerspecs, x, y, batch_size
end

# Function to run a single training iteration
function run_training_iteration(layers, x, y, batch_size, hp)
    feedforward!(layers, x, batch_size)
    backprop!(layers, y, batch_size)
    update_weight_loop!(layers, hp, 1)
end

# Main profiling function
function profile_adam()
    layerspecs, x, y, batch_size = create_test_network()
    layers = setup_train(layerspecs, batch_size)
    hp = HyperParameters(lr=0.001, reg=:none, regparm=0.0, do_stats=false)
    
    # Warm up
    for _ in 1:5
        run_training_iteration(layers, x, y, batch_size, hp)
    end
    
    # Profile
    Profile.clear()
    @profile for _ in 1:10
        run_training_iteration(layers, x, y, batch_size, hp)
    end
    
    # Print profile results
    Profile.print()
    
    # Benchmark specific functions
    println("\nBenchmarking update_weights!")
    @btime update_weights!($layers[2], $hp, 1)
    
    println("\nBenchmarking update_batchnorm!")
    @btime update_batchnorm!($layers[2], $hp, 1)
    
    println("\nBenchmarking pre_adam!")
    @btime pre_adam!($layers[2], $layers[2].optparams, 1)
    
    println("\nBenchmarking pre_adam_batchnorm!")
    @btime pre_adam_batchnorm!($layers[2].normparams, $layers[2].optparams, 1)
    
    println("\nBenchmarking adam_helper!")
    @btime adam_helper!($layers[2].grad_m_weight, $layers[2].grad_v_weight, $layers[2].grad_weight, $layers[2].optparams, 1)
end

# Run the profiling
profile_adam() 