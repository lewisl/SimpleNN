#=
TODO
- test ConvLayer with no padding
- best way to set input layer type: flat or image
- best way to set output layer type: softmax, single, multi, regression?

- implement ADAM and momentum
- test regression

=#


# ============================
# Setup training
# ============================

# method pre-declaration so we can use these functions as inputs to the struct. actual functions defined way below.
function simple_update! end
function reg_L1_update! end
function reg_L2_update! end

Base.@kwdef mutable struct HyperParameters
    lr::Float64 = 0.02          # learning rate
    reg::Symbol = :none         # one of :none, :L1, :L2
    regparm::Float64 = 0.01     # typically called lambda
    weight_update_f!::Function = simple_update!
    do_stats::Bool = true

    function HyperParameters(lr::Float64, reg::Symbol, regparm::Float64, weight_update_f!::Function, do_stats::Bool)
        if !in(reg, [:none, :L1, :L2])
            error("reg must be one of :none, :L1, :l2. Input was :$reg")
        end
        if reg == :L1
            weight_update_f! = reg_L1_update!
        elseif reg == :L2
            weight_update_f! = reg_L2_update!
        end
        new(lr, reg, regparm, weight_update_f!, do_stats)
    end
end

default_hp = HyperParameters()     # to pass defaults into training_loop



function setup_train(layerspecs::Vector{LayerSpec}, batch_size)

    layers = allocate_layers(layerspecs, batch_size)
    show_all_array_sizes(layers)

    return layers
end


function show_array_sizes(layer)
    println("name: ", layer.name)
    for p in fieldnames(typeof(layer))
        val = getfield(layer, p)
        if isa(val, AbstractArray)
            println(p,": ",size(val))
        end
    end
end


function show_all_array_sizes(layers)
    for lr in layers
        println()
        show_array_sizes(lr)
        println()
    end
    return
end

function he_initialize(weight_dims::NTuple{4,Int64}; scale=2.0, adj=0.0)
    k_h, k_w, in_channels, out_channels = weight_dims
    fan_in = k_h * k_w * in_channels
    scale_factor = scale / (1.0 + adj^2) / Float64(fan_in)
    randn(k_h, k_w, in_channels, out_channels) .* sqrt(scale_factor)
end

function he_initialize(weight_dims::NTuple{2,Int64}; scale=2.0, adj=0.0)
    # adj is typically for leaky relu
    n_in, n_out = weight_dims
    scale_factor = scale / (1.0 + adj^2) / Float64(n_in)
    randn(n_in, n_out) .* sqrt(scale_factor)
end


# ============================
# pre-alllocation of Layers
# ============================

function allocate_layers(lsvec::Vector{LayerSpec}, n_samples)

    Random.seed!(42)
    layerdat = Layer[]

    for (idx, lr) in enumerate(lsvec)
        if idx == 1
            if lr.kind != :input
                error("First layer must be the input layer.")
            else
                push!(layerdat,
                    InputLayer(
                        name=lr.name,
                        kind=lr.kind,
                        out_h=lr.h,
                        out_w=lr.w,
                        outch=lr.outch,
                        a=zeros(lr.h, lr.w, lr.outch, n_samples),
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
    stats = StatSeries(
        acc=zeros(no_of_batches * epochs),
        cost=zeros(no_of_batches * epochs),
        batch_size=batch_size,
        epochs=epochs,
        minibatch_size=minibatch_size)
end


# ============================
# Setup prediction
# ============================


function setup_preds(predlayerspecs, layers::Vector{<:Layer}, n_samples)
    predlayers = allocate_layers(predlayerspecs, n_samples)
    for (prlr, lr) in zip(predlayers, layers)
        if (typeof(prlr) == ConvLayer) | (typeof(prlr) == LinearLayer)
            prlr.weight .= lr.weight
            if prlr.dobias
                prlr.bias .= lr.bias
            end
            if prlr.normparams isa BatchNorm
                prlr.normparams.gam .= lr.normparams.gam
                prlr.normparams.bet .= lr.normparams.bet
                prlr.normparams.mu_run .= lr.normparams.mu_run
                prlr.normparams.std_run .= lr.normparams.std_run
                prlr.normparams.istraining = false
            end
        end
    end
    return predlayers
end

# this is a faster way to do argmax
function find_max_idx(arr::AbstractVector)
    max_idx = 1
    max_val = arr[1]
    @inbounds for i in (first(axes(arr,1))+1):last(axes(arr,1))
        if arr[i] > max_val
            max_val = arr[i]
            max_idx = i
        end
    end
    return max_idx
end

function accuracy_count(preds, targets)
    (correct_count, total_samples) =_accuracy_base(preds, targets, true)
end

function accuracy(preds, targets)
    (correct_count, total_samples) =_accuracy_base(preds, targets)
    correct_count / total_samples
end

function _accuracy_base(preds, targets, onlycount=false)
    # non-allocating version of accuracy without using argmax
    if size(targets, 1) > 1
        # Multi-class classification
        correct_count = 0
        total_samples = size(preds, 2)  # Assuming column-major layout (examples are columns; rows are features of an example)

        @inbounds for sample in 1:total_samples
            @views target_max_idx = find_max_idx(targets[:,sample])

            @views pred_max_idx = find_max_idx(preds[:,sample])

            if pred_max_idx == target_max_idx
                correct_count += 1
            end
        end
        return correct_count, total_samples
        
    else
        # Binary classification
        correct_count = 0
        total_samples = length(preds)

        @inbounds for i in eachindex(preds)
            is_correct = (preds[i] > 0.5) == (targets[i] > 0.5)
            correct_count += is_correct ? 1 : 0
        end
        return correct_count, total_samples
    end
end

# ============================
# cost
# ============================


function cross_entropy_cost(pred::AbstractMatrix{Float64}, target::AbstractMatrix{Float64}, n_samples)
    # this may not be obvious, but it is a non-allocating version
    n = n_samples
    log_sum1 = 0.0
    log_sum2 = 0.0

    @inbounds for i in eachindex(pred)
        # First term: target * log(max(pred, 1e-20))
        pred_val = max(pred[i], 1e-20)
        log_sum1 += target[i] * log(pred_val)

        # Second term: (1-target) * log(max(1-pred, 1e-20))
        inv_pred = max(1.0 - pred[i], 1e-20)
        log_sum2 += (1.0 - target[i]) * log(inv_pred)
    end

    return (-1.0 / n) * (log_sum1 + log_sum2)
end


function mse_cost(targets, predictions, n, theta=[], lambda=1.0, reg="", output_layer=3)
    @fastmath cost = (1.0 / (2.0 * n)) .* sum((targets .- predictions) .^ 2.0)
    @fastmath if reg == "L2"  # set reg="" if not using regularization
        regterm = lambda/(2.0 * n) .* sum([dot(th, th) for th in theta[2:output_layer]])
        cost = cost + regterm
    end
    return cost
end

# ============================
# Training Loop
# ============================

function feedforward!(layers::Vector{<:Layer}, x, n_samples)
    layers[begin].a .= x
    @inbounds for (i, lr) in zip(2:length(layers), layers[2:end])  # assumes that layers[1] MUST be input layer without checking!
        # dispatch on type of lr
        layer_forward!(lr, (layers[i-1].a), n_samples)
    end
    return
end

function backprop!(layers::Vector{<:Layer}, y, n_samples)
    # output layer is different
    dloss_dz!(layers[end], y)
    layer_backward!(layers[end], layers[end-1], n_samples; output=true)

    # skip over output layer (end) and input layer (begin)
    @inbounds @views for (j, lr) in enumerate(reverse(layers[begin+1:end-1]))
        i = length(layers) - j
        layer_backward!(lr, layers[i+1], n_samples)
    end
    return
end

######################
# update weights and optimization
######################


function simple_update!(layer::Layer, hp)
    # use explicit loop to eliminate allocation and allow optimization
    @inbounds for i in eachindex(layer.weight)
        layer.weight[i] -= hp.lr * layer.grad_weight[i]
    end

    if layer.normparams isa BatchNorm
        update_batchnorm!(layer.normparams, hp)
    end

    # Separate loop for bias
    layer.dobias && @inbounds for i in eachindex(layer.bias)
        layer.bias[i] -= hp.lr * layer.grad_bias[i]
    end
end


function update_batchnorm!(bn::BatchNorm, hp)
    @fastmath bn.gam .-= (hp.lr .* bn.delta_gam)
    @fastmath bn.bet .-= (hp.lr .* bn.delta_bet)
end

function reg_L1_update!(layer::Layer, hp)
    @inbounds for i in eachindex(layer.weight)
        layer.weight[i] -= hp.lr * (layer.grad_weight[i] + hp.regparm * sign(layer.weight[i]))
    end

    # Separate loop for bias with no regularization
    layer.dobias && @inbounds for i in eachindex(layer.bias)
        layer.bias[i] -= hp.lr * layer.grad_bias[i]
    end
end

function reg_L2_update!(layer::Layer, hp)
    @inbounds for i in eachindex(layer.weight)
        layer.weight[i] -= hp.lr * (layer.grad_weight[i] + hp.regparm * layer.weight[i])
    end

    # Separate loop for bias with no regularization
    layer.dobias && @inbounds for i in eachindex(layer.bias)
        layer.bias[i] -= hp.lr * layer.grad_bias[i]
    end
end


function update_weight_loop!(layers::Vector{<:Layer}, hp)
    for lr in layers[begin+1:end]
        if (typeof(lr) == FlattenLayer) | (typeof(lr) == MaxPoolLayer)
            continue  # redundant but obvious
        end
        hp.weight_update_f!(lr, hp)

        if lr.normparams isa BatchNorm  # test if the field is a struct of type BatchNorms
            update_batchnorm!(lr.normparams, hp)
        end

    end
end


function train_loop!(layers::Vector{L}; x, y, full_batch, epochs, minibatch_size=0, hp=default_hp) where L <: Layer

    # setup minibatches
    if minibatch_size == 0
        # setup noop minibatch parameters
        mini_num = 1
        n_samples = full_batch
    else
        if rem(full_batch, minibatch_size) != 0
            error("minibatch_size does not divide evenly into batch_size")
        else
            mini_num = div(full_batch, minibatch_size)
            n_samples = minibatch_size
        end
    end

    @show mini_num
    @show n_samples

    stats = allocate_stats(full_batch, minibatch_size, epochs)
    counter = 0

    for e = 1:epochs
        @inbounds for batno in 1:mini_num

            if mini_num > 1
                start_obs = (batno - 1) * minibatch_size + 1
                end_obs = start_obs + minibatch_size - 1
                x_part = view(x, :, :, :, start_obs:end_obs)
                y_part = view(y, :, start_obs:end_obs)
            else
                x_part = x
                y_part = y
            end

            counter += 1

            print("counter = ", counter, "\r")
            flush(stdout)
            feedforward!(layers, x_part, n_samples)

            backprop!(layers, y_part, n_samples)

            update_weight_loop!(layers, hp)

            hp.do_stats  && gather_stats!(stats, layers, y_part, counter, batno, e; to_console=false)

        end
    end

    return stats
end


"""
    gather_stats!(stat_series, layers, y_train, counter, batno, epoch; to_console=true, to_series=true)

During each training loop, add loss and accuracy to stat_series for each batch trained.
"""
function gather_stats!(stat_series, layers, y_train, counter, batno, epoch; to_console=true, to_series=true)
    cost = cross_entropy_cost((layers[end].a), y_train, (stat_series.minibatch_size))
    acc = accuracy((layers[end].a), y_train)
    if to_series
        stat_series.cost[counter] = cost
        stat_series.acc[counter] = acc
    end

    if to_console
        println("\nepoch $epoch batch $batno Loss = $loss_val Accuracy = $acc\n")
    end
end


function plot_stats(stats)
    plot(stats.acc, label="Accuracy", color=:blue, ylabel="Accuracy")
    plot!(twinx(), stats.loss, label="Cost", ylabel="Loss", color=:red)
end

function minibatch_prediction(layers::Vector{Layer}; x, y, minibatch_size=0, costfunc = cross_entropy_cost)
    (out,full_batch) = size(y)

    # setup minibatches
    if minibatch_size == 0
        # setup noop minibatch parameters
        mini_num = 1
        n_samples = full_batch
    else
        if rem(full_batch, minibatch_size) != 0
            error("minibatch_size does not divide evenly into batch_size")
        else
            mini_num = div(full_batch, minibatch_size)
            n_samples = minibatch_size
        end
    end

    total_correct = 0
    total_cnt = 0
    total_cost = 0
    # pre-allocate outcomes for use in loop
    preds = zeros(out,n_samples)
    targets = zeros(out, n_samples)

    for batno in 1:mini_num
        if mini_num > 1
            start_obs = (batno - 1) * minibatch_size + 1
            end_obs = start_obs + minibatch_size - 1
            x_part = view(x, :, :, :, start_obs:end_obs)
            y_part = view(y, :, start_obs:end_obs)
        else
            x_part = x
            y_part = y
        end

        feedforward!(layers, x_part, n_samples)

        # stats per batch: can't use x_part because that is input, layer 1
        @views preds .= layers[end].a
        targets .= y_part

        (correct_count, total_samples) = accuracy_count(preds, targets)
        cost = costfunc(preds, targets, n_samples)
        total_correct += correct_count
        total_cnt += total_samples
        total_cost += cost
    end

    return total_correct/total_cnt, total_cost/mini_num
end

function prediction(predlayers::Vector{<:Layer}, x_input, y_input)
    n_samples = size(x_input, ndims(x_input))
    feedforward!(predlayers, x_input, n_samples)
    preds = predlayers[end].a
    acc = accuracy(preds, y_input)
    cost = cross_entropy_cost(preds, y_input, n_samples)
    println("Accuracy $acc  Cost $cost")
end


# ============================
# Utility Functions
# ============================

function random_onehot(i, j)
    arr = zeros(i, j)
    for n in axes(arr, 2)
        rowselector = rand(1:10)
        arr[rowselector, n] = 1.0
    end
    return arr
end

# ============================
# Save and load weights to/from files
# ============================

function weights2file(layers, suffix, pathstr)
    for lr in fieldnames(typeof(layers))
        serialize(joinpath(pathstr, string(lr) * "_bias" * '_' * suffix * ".dat"),
            eval(getfield(layers, lr)).bias)
        serialize(joinpath(pathstr, string(lr) * "_weight" * '_' * suffix * ".dat"),
            eval(getfield(layers, lr)).weight)
    end
end

# TODO this can't work any more
# function file2weights(suffix, pathstr)
#     outlayers = init_layers(n_samples=batch_size)
#     for lr in fieldnames(typeof(outlayers))
#         fname_bias = joinpath(pathstr, string(lr) * "_bias" * '_' * suffix * ".dat")
#         fname_weight = joinpath(pathstr, string(lr) * "_weight" * '_' * suffix * ".dat")
#         setfield!(getfield(layers, lr), :bias, deserialize(fname_bias))
#         setfield!(getfield(layers, lr), :weight, deserialize(fname_weight))
#     end
#     return outlayers
# end
