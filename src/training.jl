#=
TODO

- test regression
- make stats struct immutable
- ADAMW not working with reg=:none


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

function allocate_layers(lsvec::Vector{LayerSpec}, n_samples)

    Random.seed!(42)
    layerdat = Layer[]

    for (idx, lr) in enumerate(lsvec)
        if idx == 1
            if lr.kind != :input
                error("First layer must be the input layer.")
            else
                push!(layerdat,
                    InputLayer(lr, n_samples))
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

function allocate_stats(full_batch, minibatch_size, epochs)
    no_of_batches = div(full_batch, minibatch_size)
    stats = StatSeries(
        acc=zeros(ELT, no_of_batches * epochs),
        cost=zeros(ELT, no_of_batches * epochs),
        batch_size=full_batch,
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


function cross_entropy_cost(pred::AbstractMatrix{ELT}, target::AbstractMatrix{ELT}, n_samples)
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


function mse_cost(targets, predictions, n, theta=[], lambda=ELT(1.0), reg="", output_layer=3)
    @fastmath cost = (ELT(1.0) / (ELT(2.0) * n)) .* sum((targets .- predictions) .^ ELT(2.0))
    @fastmath if reg == "L2"  # set reg="" if not using regularization
        regterm = lambda / (ELT(2.0) * n) .* sum([dot(th, th) for th in theta[2:output_layer]])
        cost = cost + regterm
    end
    return cost
end

# ============================
# Training Loop
# ============================

function feedforward!(layers::Vector{<:Layer}, x)
    layers[begin].a .= x
    @inbounds for (i, lr) in zip(2:length(layers), layers[2:end])  # assumes that layers[1] MUST be input layer without checking!        
        lr(layers[i-1].a)  # dispatch on type of lr
    end
    return
end

function backprop!(layers::Vector{<:Layer}, y)
    # output layer is different
    dloss_dz!(layers[end], y)
    layers[end](layers[end-1])

    # skip over output layer (end) and input layer (begin)
    nlayers = length(layers)
    @inbounds @views for (i, lr) in zip((nlayers-1):-1:2, reverse(layers[begin+1:end-1]))
        lr(layers[i+1])
    end
    return
end

######################
# update weights and optimization
######################


function update_weights!(layer::Layer, hp, t)
    # Handle Adam/AdamW optimization based on layer's optparams
    if isa(layer.optparams, AdamParam)
        ad = layer.optparams
        pre_adam!(layer, ad, t)
        b1_divisor = ELT(1.0) / (ELT(1.0) - ad.b1^t)
        b2_divisor = ELT(1.0) / (ELT(1.0) - ad.b2^t)

        # Update weights with @turbo for better performance
        @turbo for i in eachindex(layer.weight)
            # Base Adam update term
            adam_term = (layer.grad_m_weight[i] * b1_divisor) / (sqrt(layer.grad_v_weight[i] * b2_divisor) + IT)
            layer.weight[i] -= hp.lr * adam_term
        end

        # Apply regularization and weight decay outside @turbo
        if hp.reg == :L1
            @turbo for i in eachindex(layer.weight)
                layer.weight[i] -= hp.lr * hp.regparm * sign(layer.weight[i])
            end
        elseif hp.reg == :L2
            @turbo for i in eachindex(layer.weight)
                layer.weight[i] -= hp.lr * hp.regparm * layer.weight[i]
            end
        end

        # Apply weight decay if needed
        if ad.decay > 0
            @turbo for i in eachindex(layer.weight)
                layer.weight[i] -= hp.lr * ad.decay * layer.weight[i]
            end
        end

        # Update biases with @turbo
        if layer.dobias
            @turbo for i in eachindex(layer.bias)
                adam_term = (layer.grad_m_bias[i] * b1_divisor) / (sqrt(layer.grad_v_bias[i] * b2_divisor) + IT)
                layer.bias[i] -= hp.lr * adam_term
            end
        end

        # Update batch normalization parameters if present
        if isa(layer.normparams, BatchNorm)
            bn = layer.normparams
            pre_adam_batchnorm!(bn, ad, t)

            # Update gamma (scale) parameter and beta (shift parameter)
            @turbo for i in eachindex(bn.gam)
                adam_term_gam = (bn.grad_m_gam[i] * b1_divisor) / (sqrt(bn.grad_v_gam[i] * b2_divisor) + IT)
                bn.gam[i] -= hp.lr * adam_term_gam

                adam_term_bet = (bn.grad_m_bet[i] * b1_divisor) / (sqrt(bn.grad_v_bet[i] * b2_divisor) + IT)
                bn.bet[i] -= hp.lr * adam_term_bet
            end
        end
    else
        # Simple SGD update
        @turbo for i in eachindex(layer.weight)
            layer.weight[i] -= hp.lr * layer.grad_weight[i]
        end

        # Apply regularization outside @turbo
        if hp.reg == :L1
            @turbo for i in eachindex(layer.weight)
                layer.weight[i] -= hp.lr * hp.regparm * sign(layer.weight[i])
            end
        elseif hp.reg == :L2
            @turbo for i in eachindex(layer.weight)
                layer.weight[i] -= hp.lr * hp.regparm * layer.weight[i]
            end
        end

        # Simple SGD update if not using Adam/AdamW
        if layer.normparams isa BatchNorm
            bn = layer.normparams
            @turbo for i in eachindex(bn.gam)
                bn.gam[i] -= hp.lr * bn.grad_gam[i]
                bn.bet[i] -= hp.lr * bn.grad_bet[i]
            end
        end

        if layer.dobias
            @turbo for i in eachindex(layer.bias)
                layer.bias[i] -= hp.lr * layer.grad_bias[i]
            end
        end
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

# helpers for minibatch views in training loop
@inline function slice_minibatch(x::AbstractArray{T,2}, range) where T
    view(x, :, range)
end

@inline function slice_minibatch(x::AbstractArray{T,4}, range) where T  
    view(x, :, :, :, range)
end


function train!(layers::Vector{L}; x, y, full_batch, epochs, minibatch_size=0, hp=default_hp) where {L<:Layer}

    dobatch = if minibatch_size == 0
                false
            elseif minibatch_size <= 39
                error("Minibatch_size too small.  Choose a larger minibatch_size.")
            elseif full_batch / minibatch_size > 3
                true
            else
                error("Minibatch_size too large with fewer than 3 batches. Choose a much smaller minibatch_size.")
            end

    stats = allocate_stats(full_batch, minibatch_size, epochs)
    batch_counter = 0

    for e = 1:epochs
        samples_left = full_batch
        start_obs = end_obs = 0
        loop = true

        @inbounds while loop 

            if dobatch
                if samples_left > minibatch_size  # continue
                    start_obs = end_obs + 1
                    end_obs = start_obs + minibatch_size - 1
                else   # stop after this iteration setting the stop flag
                    start_obs = end_obs + 1
                    end_obs = start_obs + samples_left - 1
                    loop = false
                end
                x_part = slice_minibatch(x, start_obs:end_obs)
                y_part = slice_minibatch(y, start_obs:end_obs)
                samples_left -= minibatch_size  # update the effective loop counter
            else
                x_part = x
                y_part = y
                loop = false  # just do it once per epoch
            end

            batch_counter += 1

            print("counter = ", batch_counter, "\r")
            flush(stdout)

            feedforward!(layers, x_part)

            backprop!(layers, y_part)

            update_weight_loop!(layers, hp, batch_counter)

            hp.do_stats && gather_stats!(stats, layers, y_part, batch_counter, batno, e; to_console=false)

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
        println("\nepoch $epoch batch $batno Loss = $cost Accuracy = $acc\n")
    end
end


function plot_stats(stats)
    plot(stats.acc, label="Accuracy", color=:blue, ylabel="Accuracy")
    plot!(twinx(), stats.loss, label="Cost", ylabel="Loss", color=:red)
end

function minibatch_prediction(layers::Vector{Layer}, x, y, costfunc=cross_entropy_cost)
    (out, full_batch) = size(y)
    minibatch_size = size(layers[end].a, 2)  # TODO is this always going to work?

    # pre-allocate outcomes for use in loop
    preds = zeros(ELT, out, minibatch_size)
    targets = zeros(ELT, out, minibatch_size)

    total_correct = 0
    total_cnt = 0
    total_cost = 0
    samples_left = full_batch
    start_obs = end_obs = 0
    loop = true
    batch_counter = 0

    @inbounds while loop
        if samples_left > minibatch_size  # continue
            start_obs = end_obs + 1
            end_obs = start_obs + minibatch_size - 1
            n_samples = minibatch_size
        else   # stop after this iteration setting the stop flag
            start_obs = end_obs + 1
            end_obs = start_obs + samples_left - 1
            loop = false
            n_samples = samples_left
        end
        x_part = slice_minibatch(x, start_obs:end_obs)
        y_part = slice_minibatch(y, start_obs:end_obs)
        samples_left -= minibatch_size  # update the effective loop counter

        batch_counter += 1

        feedforward!(layers, x_part)

        # stats per batch: can't use x_part because that is input, layer 1
        @views preds .= layers[end].a
        targets .= y_part

        (correct_count, total_samples) = accuracy_count(preds, targets)
        cost = costfunc(preds, targets, n_samples)
        total_correct += correct_count
        total_cnt += total_samples
        total_cost += cost
    end

    return total_correct / total_cnt, total_cost / batch_counter
end

function prediction(predlayers::Vector{<:Layer}, x_input, y_input)
    n_samples = size(x_input, ndims(x_input))
    feedforward!(predlayers, x_input)
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
