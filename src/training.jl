#=
TODO

- make stats struct immutable
- test logistic as sigmoid activation and gradient
- use mse_cost for regression
- implement prediction for regression
- implement anova for regression
- is there any real point to using regularization in the cost
    calculation when cost is not directly used in backprop?
    A: only if printing cost progress during training and for training progress
        statistics
- hide the minibatch complexity in a function
- split apart update weights mess


=#

using LoopVectorization

# ============================
# Setup training
# ============================


Base.@kwdef mutable struct HyperParameters
    lr::ELT = ELT(0.02)        # learning rate
    reg::Symbol = :none      # one of :none, :L1, :L2
    regparm::ELT = ELT(0.0004)  # typically called lambda
    do_stats::Bool = true

    function HyperParameters(lr::ELT, reg::Symbol, regparm::ELT, do_stats::Bool)
        if !in(reg, [:none, :L1, :L2])
            error("reg must be one of :none, :L1, :L2. Input was :$reg")
        end
        new(lr, reg, regparm, do_stats)
    end
end

default_hp = HyperParameters()     # to pass defaults into training_loop

Base.@kwdef mutable struct StatSeries
    acc::Array{ELT,1} = ELT[]
    cost::Array{ELT,1} = ELT[]
    batch_size::Int = 0
    epochs::Int = 0
    minibatch_size = 0
end

function setup_train(layerspecs::Vector{LayerSpec}, batch_size)

    layers = allocate_layers(layerspecs, batch_size)
    show_layer_specs(layers)

    return layers
end


function show_array_sizes(layer)
    println("name: ", layer.name)
    println("kind: ", typeof(layer))
    println("arrays:")
    for p in fieldnames(typeof(layer))
        val = getfield(layer, p)
        if isa(val, AbstractArray)
            println(p, ": ", size(val))
        end
    end
end

function show_functions(layer)
    first = true
    for p in fieldnames(typeof(layer))
        val = getfield(layer, p)
        if isa(val, Function)
            first && println("functions:")
            println(p, ": ", val, " ", typeof(val))
            first = false
        end
    end
end

function show_layer_specs(layers)
    for lr in layers
        println()

        show_array_sizes(lr)
        show_functions(lr)
        println()
    end
    return
end

function he_initialize(weight_dims::NTuple{4,Int64}; scale=ELT(2.0), adj=ELT(0.0))
    k_h, k_w, in_channels, out_channels = weight_dims
    fan_in = k_h * k_w * in_channels
    scale_factor = scale / (ELT(1.0) + adj^2) / ELT(fan_in)
    randn(k_h, k_w, in_channels, out_channels) .* sqrt(scale_factor)
end

function he_initialize(weight_dims::NTuple{2,Int64}; scale=ELT(2.0), adj=ELT(0.0))
    # adj is typically for leaky relu
    n_in, n_out = weight_dims
    scale_factor = scale / (ELT(1.0) + adj^2) / ELT(n_in)
    randn(n_in, n_out) .* sqrt(scale_factor)
end


# ============================
# pre-alllocation of Layers
# ============================

function allocate_layers(lsvec::Vector{LayerSpec}, batch_size)

    Random.seed!(42)
    layerdat = Layer[]

    for (idx, lr) in enumerate(lsvec)
        if idx == 1
            if lr.kind != :input
                error("First layer must be the input layer.")
            else
                push!(layerdat,
                    InputLayer(lr, batch_size))
            end
            continue  # skip the ifs and go to next lr
        elseif lr.kind == :conv
            push!(layerdat, ConvLayer(lr, layerdat[idx-1], batch_size))
        elseif lr.kind == :linear
            push!(layerdat, LinearLayer(lr, layerdat[idx-1], batch_size))
        elseif lr.kind == :maxpool
            push!(layerdat, MaxPoolLayer(lr, layerdat[idx-1], batch_size))
        elseif lr.kind == :flatten
            push!(layerdat, FlattenLayer(lr, layerdat[idx-1], batch_size))
        else
            error("Found unrecognized layer kind")
        end
    end

    return layerdat
end


function allocate_stats(fullbatch, minibatch_size, epochs)
    no_of_batches = div(fullbatch, minibatch_size)
    stats = StatSeries(
        acc=zeros(ELT, no_of_batches * epochs),
        cost=zeros(ELT, no_of_batches * epochs),
        batch_size=fullbatch,
        epochs=epochs,
        minibatch_size=minibatch_size)
end


# ============================
# Setup prediction
# ============================


function setup_preds(predlayerspecs, layers::Vector{<:Layer}, n_samples)
    predlayers = allocate_layers(predlayerspecs, n_samples)
    for (prlr, lr) in zip(predlayers, layers)     # TODO are we missing any other trainable parameters?
        if isa(prlr, ConvLayer) | isa(prlr, LinearLayer)
            prlr.weight .= lr.weight
            if prlr.dobias
                prlr.bias .= lr.bias
            end
            if prlr.normparams isa BatchNorm
                prlr.normparams.gam .= lr.normparams.gam
                prlr.normparams.bet .= lr.normparams.bet
                prlr.normparams.mu_run .= lr.normparams.mu_run
                prlr.normparams.std_run .= lr.normparams.std_run
                prlr.normparams.istraining[] = false   # setting value of Ref field (like a 1 element array)
            end
        end
    end
    return predlayers
end

# this is a faster way to do argmax
function find_max_idx(arr::AbstractVector)
    max_idx = 1
    max_val = arr[1]
    @inbounds for i in (first(axes(arr, 1))+1):last(axes(arr, 1))
        if arr[i] > max_val
            max_val = arr[i]
            max_idx = i
        end
    end
    return max_idx
end

function accuracy_count(preds, targets)
    (correct_count, total_samples) = _accuracy_base(preds, targets, true)
end

function accuracy(preds, targets)
    (correct_count, total_samples) = _accuracy_base(preds, targets)
    correct_count / total_samples
end

function _accuracy_base(preds, targets, onlycount=false)
    # non-allocating version of accuracy without using argmax
    if size(targets, 1) > 1
        # Multi-class classification
        correct_count = 0
        total_samples = size(preds, 2)  # Assuming column-major layout (examples are columns; rows are features of an example)

        @inbounds for sample in 1:total_samples
            @views target_max_idx = find_max_idx(targets[:, sample])

            @views pred_max_idx = find_max_idx(preds[:, sample])

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


function cross_entropy_cost(pred::AbstractMatrix{ELT}, target::AbstractMatrix{ELT}, n_samples, training=false)
    # this may not be obvious, but it is a non-allocating version
    n = n_samples
    log_sum1 = ELT(0.0)
    log_sum2 = ELT(0.0)

    @inbounds for i in eachindex(pred)
        # First term: target * log(max(pred, 1e-20))
        pred_val = max(pred[i], IT)
        log_sum1 += target[i] * log(pred_val)

        # Second term: (1-target) * log(max(1-pred, 1e-20))
        inv_pred = max(ELT(1.0) - pred[i], IT)
        log_sum2 += (ELT(1.0) - target[i]) * log(inv_pred)
    end

    return (ELT(-1.0) / n) * (log_sum1 + log_sum2)
end


function mse_cost(targets, predictions, n, istraining = false, theta=[], lambda=ELT(1.0), reg="", output_layer=2)
    @fastmath cost = (ELT(1.0) / (ELT(2.0) * n)) .* sum((targets .- predictions) .^ ELT(2.0))
    if istraining
        @fastmath if reg == "L2"  # set reg="" if not using regularization
            regterm = lambda / (ELT(2.0) * n) .* sum([dot(th, th) for th in theta[2:output_layer]])
            cost = cost + regterm
        end
    end
    return cost
end

# ============================
# Training Loop
# ============================

function feedforward!(layers::Vector{<:Layer}, x, current_batch_size)
    cb = current_batch_size
    cb_rng = 1:cb

    # @show size(layers[begin].a)


    va = view_minibatch(layers[begin].a, cb_rng)

    # @show size(va)
    # @show size(x)

    @turbo va .= x
    @inbounds for (i, lr) in zip(2:length(layers), layers[2:end])  # assumes that layers[1] MUST be input layer without checking!   
        # lr.mb_range[] = mb_range    # update the layer's value for minibatch range
        lr(layers[i-1].a, cb)  # dispatch on type of lr
    end
    return
end

function backprop!(layers::Vector{<:Layer}, y, current_batch_size)
    cb = current_batch_size
    # output layer is different
    dloss_dz!(layers[end], y, cb)
    # layers[end].mb_range[] = mb_range  # update the layer's value for minibatch range
    layers[end](layers[end-1], cb)   # calls layer function for backward pass, passes layers[end] and layers[end-1]

    nlayers = length(layers)  # skip over output layer (end) and input layer (begin)
    @inbounds @views for (i, lr) in zip((nlayers-1):-1:2, reverse(layers[begin+1:end-1]))
        # lr.mb_range[] = mb_range   # update the layer's value for minibatch range
        lr(layers[i+1], current_batch_size)
    end
    return
end

######################
# update weights and optimization
######################


function update_weights!(layer::Layer, hp, t)
    if isa(layer.optparams, AdamParam)
        # Core optimizer updates - all Adam-specific logic moved into helpers
        update_weights_adam!(layer, hp, t)
        update_bias_adam!(layer, hp, t)
        update_batchnorm_adam!(layer, hp, t)
        
        # Regularization and weight decay
        apply_regularization!(layer, hp)
        apply_weight_decay!(layer, hp, layer.optparams.decay)
    else
        # Core SGD updates
        update_weights_sgd!(layer, hp)
        update_bias_sgd!(layer, hp)
        update_batchnorm_sgd!(layer, hp)
        
        # Regularization
        apply_regularization!(layer, hp)
    end
end

function update_weight_loop!(layers::Vector{<:Layer}, hp, counter)
    for lr in layers[begin+1:end]
        if isa(lr, FlattenLayer) || isa(lr, MaxPoolLayer)
            continue  # skip non-parametric layers
        end

        update_weights!(lr, hp, counter)
    end
end


# ============================
# Helper functions for regularization and weight decay
# ============================

function apply_regularization!(layer::Layer, hp)
    if hp.reg == :L1
        @turbo for i in eachindex(layer.weight)
            layer.weight[i] -= hp.lr * hp.regparm * sign(layer.weight[i])
        end
    elseif hp.reg == :L2
        @turbo for i in eachindex(layer.weight)
            layer.weight[i] -= hp.lr * hp.regparm * layer.weight[i]
        end
    end
end

function apply_weight_decay!(layer::Layer, hp, decay)
    if decay > 0
        @turbo for i in eachindex(layer.weight)
            layer.weight[i] -= hp.lr * decay * layer.weight[i]
        end
    end
end

# ============================
# Helper functions for Adam optimizer
# ============================

function update_weights_adam!(layer::Layer, hp, t)
    ad = layer.optparams
    pre_adam!(layer, ad, t)
    b1_divisor = ELT(1.0) / (ELT(1.0) - ad.b1^t)
    b2_divisor = ELT(1.0) / (ELT(1.0) - ad.b2^t)
    
    @turbo for i in eachindex(layer.weight)
        layer.weight[i] -= hp.lr * ((layer.grad_m_weight[i] * b1_divisor) / (sqrt(layer.grad_v_weight[i] * b2_divisor) + IT))
    end
end

function update_bias_adam!(layer::Layer, hp, t)
    layer.dobias || return  # Early return if no bias
    
    ad = layer.optparams
    b1_divisor = ELT(1.0) / (ELT(1.0) - ad.b1^t)
    b2_divisor = ELT(1.0) / (ELT(1.0) - ad.b2^t)
    
    @turbo for i in eachindex(layer.bias)
        adam_term = (layer.grad_m_bias[i] * b1_divisor) / (sqrt(layer.grad_v_bias[i] * b2_divisor) + IT)
        layer.bias[i] -= hp.lr * adam_term
    end
end

function update_batchnorm_adam!(layer::Layer, hp, t)
    isa(layer.normparams, BatchNorm) || return  # Early return if no batch norm
    
    ad = layer.optparams
    bn = layer.normparams
    pre_adam_batchnorm!(bn, ad, t)
    b1_divisor = ELT(1.0) / (ELT(1.0) - ad.b1^t)
    b2_divisor = ELT(1.0) / (ELT(1.0) - ad.b2^t)
    
    @turbo for i in eachindex(bn.gam)
        adam_term_gam = (bn.grad_m_gam[i] * b1_divisor) / (sqrt(bn.grad_v_gam[i] * b2_divisor) + IT)
        bn.gam[i] -= hp.lr * adam_term_gam
        
        adam_term_bet = (bn.grad_m_bet[i] * b1_divisor) / (sqrt(bn.grad_v_bet[i] * b2_divisor) + IT)
        bn.bet[i] -= hp.lr * adam_term_bet
    end
end

# ============================
# Helper functions for SGD optimizer
# ============================

function update_weights_sgd!(layer::Layer, hp)
    @turbo for i in eachindex(layer.weight)
        layer.weight[i] -= hp.lr * layer.grad_weight[i]
    end
end

function update_bias_sgd!(layer::Layer, hp)
    layer.dobias || return  # Early return if no bias
    
    @turbo for i in eachindex(layer.bias)
        layer.bias[i] -= hp.lr * layer.grad_bias[i]
    end
end

function update_batchnorm_sgd!(layer::Layer, hp)
    isa(layer.normparams, BatchNorm) || return  # Early return if no batch norm
    
    bn = layer.normparams
    @turbo for i in eachindex(bn.gam)
        bn.gam[i] -= hp.lr * bn.grad_gam[i]
        bn.bet[i] -= hp.lr * bn.grad_bet[i]
    end
end

# ============================
# update weights and optimization
# ============================

function train!(layers::Vector{L}; x, y, fullbatch, epochs, minibatch_size=0, hp=default_hp) where {L<:Layer}

        if minibatch_size == fullbatch
            dobatch = false   # we are training on the full batch
            x_part = x
            y_part = y
            current_batch_size = fullbatch
        elseif minibatch_size <= 39
            error("Minibatch_size too small.  Choose a larger minibatch_size.")
        elseif fullbatch / minibatch_size > 3
            dobatch = true
        else
            error("Minibatch_size too large with fewer than 3 batches. Choose a much smaller minibatch_size.")
        end

    hp.do_stats && (stats = allocate_stats(fullbatch, minibatch_size, epochs))  # TODO this won't work with full batch training

    eval_counter = 0
    @inbounds for e = 1:epochs
        for i in 1:minibatch_size:fullbatch
            if dobatch  # select x_part, y_part, calculate current_batch_size
                start_idx = i
                end_idx = min(i + minibatch_size - 1, fullbatch)
                mb_range = start_idx:end_idx
                x_part = view_minibatch(x, mb_range)
                y_part = view_minibatch(y, mb_range)
                current_batch_size = end_idx - start_idx + 1
            end

            eval_counter += 1

            print("counter = ", eval_counter, "\r")
            flush(stdout)

            feedforward!(layers, x_part, current_batch_size)

            backprop!(layers, y_part, current_batch_size)

            update_weight_loop!(layers, hp, eval_counter)

            hp.do_stats && (gather_stats!(stats, layers, y_part, eval_counter, batno, e; to_console=false))

        end
    end

    hp.do_stats && return stats
    return
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
        println("\nepoch $epoch batch $batno Loss = $cost Accuracy = $acc\n")
    end
end


function plot_stats(stats)
    plot(stats.acc, label="Accuracy", color=:blue, ylabel="Accuracy")
    plot!(twinx(), stats.loss, label="Cost", ylabel="Loss", color=:red)
end

function minibatch_prediction(layers::Vector{Layer}, x, y, costfunc=cross_entropy_cost)
    (out, fullbatch) = size(y)
    minibatch_size = size(layers[end].a, 2)  # TODO is this always going to work?

    # pre-allocate outcomes for use in loop
    preds = zeros(ELT, out, minibatch_size)
    targets = zeros(ELT, out, minibatch_size)

    # accumulators
    total_correct = 0
    total_cnt = 0
    total_cost = 0
    eval_counter = 0

    @inbounds for i in 1:minibatch_size:fullbatch
        start_idx = i
        end_idx = min(i + minibatch_size - 1, fullbatch)
        mb_range = start_idx:end_idx
        x_part = view_minibatch(x, mb_range)
        y_part = view_minibatch(y, mb_range)
        current_batch_size = end_idx - start_idx + 1

        eval_counter += 1

        feedforward!(layers, x_part, current_batch_size)

        # stats per batch: can't use x_part because that is input, layer 1
        preds = view_minibatch(layers[end].a, 1:current_batch_size)
        @turbo targets .= y_part

        (correct_count, total_samples) = accuracy_count(preds, targets)
        cost = costfunc(preds, targets, current_batch_size)
        total_correct += correct_count
        total_cnt += total_samples
        total_cost += cost
    end

    return total_correct / total_cnt, total_cost / eval_counter
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
    arr = zeros(ELT, i, j)
    for n in axes(arr, 2)
        rowselector = rand(1:10)
        arr[rowselector, n] = ELT(1.0)
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

# ============================
# update weights and optimization
# ============================

# helpers for minibatch views in training loop: works when range is an Int or a UnitRange{Int}
@inline function view_minibatch(x::AbstractArray{T,2}, range) where T
    # @show size(x)
    # @show range
    view(x, :, range)
end

@inline function view_minibatch(x::AbstractArray{T,4}, range) where T  
    view(x, :, :, :, range)
end
