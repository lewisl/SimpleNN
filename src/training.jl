#=
TODO
- test ConvLayer with no padding
- why don't we do implicit flatten when conv precedes linear?
- best way to set input layer type: flat or image
- best way to set output layer type: softmax, single, multi, regression?

- add regression
- add batch normalization
- add regularization
    - add reg term to cost
    - add weight reg term to delta_weight

- add ADAM optimization of learning


=#


# ============================
# Setup training
# ============================


# method to pass a function
function preptrain(modelspecs::Function, batch_size, minibatch_size)
    layerspecs = modelspecs()
    preptrain(layerspecs, batch_size, minibatch_size)
end

# method to pass a vector
function preptrain(layerspecs::Vector{LayerSpec}, batch_size, minibatch_size; preptest=false)
    trainset = MNIST(:train)
    testset = MNIST(:test)

    x_train = trainset.features[1:28, 1:28, 1:batch_size]
    x_train = Float64.(x_train)
    x_train = reshape(x_train, 28, 28, 1, batch_size)

    y_train = trainset.targets[1:batch_size]
    y_train = indicatormat(y_train)
    y_train = Float64.(y_train)

    preptest && begin
        x_test = testset.features[1:28, 1:28, :]
        x_test = Float64.(x_test)
        x_test = reshape(x_test, 28, 28, 1, :)

        y_test = testset.targets
        y_test = indicatormat(y_test)
        y_test = Float64.(y_test)
    end

    # shuffle the variables and outcome identically
    img_idx = shuffle(1:size(x_train, 4))
    x_train_shuf = x_train[:, :, :, img_idx]
    y_train_shuf = y_train[:, img_idx]

    layers = allocate_layers(layerspecs, minibatch_size)
    show_all_array_sizes(layers)

    # preptest == false
    preptest || return layerspecs, layers, x_train_shuf, y_train_shuf

    # preptest == true
    preptest && return layerspecs, layers, x_train_shuf, y_train_shuf, x_test, y_test
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
    layerdat = []

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
    stats = stat_series(
        acc=zeros(no_of_batches * epochs),
        loss=zeros(no_of_batches * epochs),
        batch_size=batch_size,
        epochs=epochs,
        minibatch_size=minibatch_size)
end


# ============================
# Setup prediction
# ============================

function setup_preds(modelspecs::Function, layers, n_samples)
    predlayerspecs = modelspecs()
    setup_preds(predlayerspecs, layers, n_samples)
end


function setup_preds(predlayerspecs, layers, n_samples)
    predlayers = allocate_layers(predlayerspecs, n_samples)
    for (prlr, lr) in zip(predlayers, layers)
        if (typeof(prlr) == ConvLayer) | (typeof(prlr) == LinearLayer)
            prlr.weight .= lr.weight
            prlr.bias .= lr.bias
        end
    end
    return predlayers
end



function accuracy(preds, targets)
    # non-allocating version of accuracy without using argmax
    if size(targets, 1) > 1
        # Multi-class classification
        correct_count = 0
        total_samples = size(preds, 2)  # Assuming column-major layout (samples in columns)

        @inbounds for sample in 1:total_samples
            # Find max index in target for this sample
            target_max_idx = 1
            target_max_val = targets[1, sample]
            @inbounds for i in (first(axes(targets, 1))+1):last(axes(targets, 1))
                if targets[i, sample] > target_max_val
                    target_max_val = targets[i, sample]
                    target_max_idx = i
                end
            end

            # Find max index in prediction for this sample
            pred_max_idx = 1
            pred_max_val = preds[1, sample]
            @inbounds for i in (first(axes(preds, 1))+1):last(axes(preds, 1))
                if preds[i, sample] > pred_max_val
                    pred_max_val = preds[i, sample]
                    pred_max_idx = i
                end
            end

            # Increment if they match
            if pred_max_idx == target_max_idx
                correct_count += 1
            end
        end

        return correct_count / total_samples
    else
        # Binary classification
        correct_count = 0
        total_samples = length(preds)

        @inbounds for i in eachindex(preds)
            is_correct = (preds[i] > 0.5) == (targets[i] > 0.5)
            correct_count += is_correct ? 1 : 0
        end

        return correct_count / total_samples
    end
end

function getvalidx(arr, argfunc=argmax)  # could also be argmin
    return vec(map(x -> x[1], argfunc(arr, dims=1)))
end



# ============================
# Training Loop
# ============================

function feedforward!(layers, x_train, n_samples)
    layers[begin].a = x_train
    @inbounds for (i, lr) in enumerate(layers[2:end])  # assumes that layers[1] MUST be input layer without checking!
        i += 1  # skip index of layer 1
        # dispatch on type of lr
        # activation function is part of the layer definition
        # @show typeof(lr)
        layer_forward!(lr, (layers[i-1].a), n_samples)
    end
    return
end

function backprop!(layers, y_train, n_samples)
    # output layer is different
    dloss_dz!(layers[end], y_train)
    layer_backward!(layers[end], layers[end-1], n_samples; output=true)

    # skip over output layer (end) and input layer (begin)
    @inbounds @views for (j, lr) in enumerate(reverse(layers[begin+1:end-1]))
        i = length(layers) - j
        layer_backward!(lr, layers[i+1], n_samples)
    end
    return
end

function weight_update!(layer, lr)
    # use explicit loop to eliminate allocation and allow optimization
    @inbounds for i in eachindex(layer.weight)
        layer.weight[i] -= lr * layer.grad_weight[i]
    end

    # Separate loop for bias
    @inbounds for i in eachindex(layer.bias)
        layer.bias[i] -= lr * layer.grad_bias[i]
    end

end

function update_weight_loop!(layers, lambda)
    @views for lr in layers[begin+1:end]
        if (typeof(lr) == FlattenLayer) | (typeof(lr) == MaxPoolLayer)
            continue  # redundant but obvious
        end
        weight_update!(lr, lambda)
    end
end


function train_loop!(layers; x_train, y_train, batch_size, epochs, minibatch_size=0, lr=0.01)

    # setup minibatches
    if minibatch_size == 0
        # setup noop minibatch parameters
        mini_num = 1
        n_samples = batch_size
    else
        if rem(batch_size, minibatch_size) != 0
            error("minibatch_size does not divide evenly into batch_size")
        else
            mini_num = div(batch_size, minibatch_size)
            n_samples = minibatch_size
        end
    end

    @show mini_num
    @show n_samples

    stats = allocate_stats(batch_size, minibatch_size, epochs)
    counter = 0

    for e = 1:epochs
        @inbounds for batno in 1:mini_num

            if mini_num > 1
                start_obs = (batno - 1) * minibatch_size + 1
                end_obs = start_obs + minibatch_size - 1
                x_train_part = view(x_train, :, :, :, start_obs:end_obs)
                y_train_part = view(y_train, :, start_obs:end_obs)
            else
                x_train_part = x_train
                y_train_part = y_train
            end

            # @show size(x_train_part)
            # @show size(y_train_part)
            counter += 1

            @show counter
            feedforward!(layers, x_train_part, n_samples)

            backprop!(layers, y_train_part, n_samples)

            update_weight_loop!(layers, lr)

            gather_stats!(stats, layers, y_train_part, counter, batno, e; to_console=false)

        end
    end

    return stats
end


"""
    gather_stats!(stat_series, layers, y_train, counter, batno, epoch; to_console=true, to_series=true)

During each training loop, add loss and accuracy to stat_series for each batch trained.
"""
function gather_stats!(stat_series, layers, y_train, counter, batno, epoch; to_console=true, to_series=true)
    loss_val = cross_entropy_loss((layers[end].a), y_train, (stat_series.minibatch_size))
    acc = accuracy((layers[end].a), y_train)
    if to_series
        stat_series.loss[counter] = loss_val
        stat_series.acc[counter] = acc
    end

    if to_console
        println("\nepoch $epoch batch $batno Loss = $loss_val Accuracy = $acc\n")
    end
end

function prediction(predlayers, x_input, y_input)
    n_samples = size(x_input, ndims(x_input))
    feedforward!(predlayers, x_input, n_samples)
    preds = predlayers[end].a
    acc = accuracy(preds, y_input)
    cost = cross_entropy_loss(preds, y_input, n_samples)
    println("Accuracy $acc  Cost $cost")
end


# ============================
# Utility Functions
# ============================

function display_mnist_digit(digit_data, dims=[])
    if length(dims) == 0
        xside = yside = convert(Int, (sqrt(length(digit_data))))
    elseif length(dims) == 1
        xside = yside = dims[1]
    elseif length(dims) >= 2
        xside = dims[2]
        yside = dims[1]
    end
    plot(Gray.(transpose(reshape(digit_data, xside, yside))), interpolation="nearest", showaxis=false)
end

function random_onehot(i, j)
    arr = zeros(i, j)
    for n in axes(arr, 2)
        rowselector = rand(1:10)
        arr[rowselector, n] = 1.0
    end
    return arr
end

function plot_training(stats)
    plot(stats.acc, label="Accuracy", color=:blue, ylabel="Accuracy")
    plot!(twinx(), stats.loss, label="Cost", ylabel="Loss", color=:red)
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

function file2weights(suffix, pathstr)
    outlayers = init_layers(n_samples=batch_size)
    for lr in fieldnames(typeof(outlayers))
        fname_bias = joinpath(pathstr, string(lr) * "_bias" * '_' * suffix * ".dat")
        fname_weight = joinpath(pathstr, string(lr) * "_weight" * '_' * suffix * ".dat")
        setfield!(getfield(layers, lr), :bias, deserialize(fname_bias))
        setfield!(getfield(layers, lr), :weight, deserialize(fname_weight))
    end
    return outlayers
end

# ============================
# Functions for performance and memory testing
# ============================

function flatloop(arr1)
    flatdim = size(arr1, 1) * size(arr1, 2) * size(arr1, 3)
    ret = zeros(flatdim, size(arr1, 4))
    innerdim = 0
    outerdim = 0
    for b in axes(arr1, 4)
        outerdim += 1
        innerdim = 0
        for c in axes(arr1, 3)
            for j in axes(arr1, 2)
                for i in axes(arr1, 1)
                    innerdim += 1
                    ret[innerdim, outerdim] = arr1[i, j, c, b]
                end
            end
        end
    end
    return ret
end

function flatloop!(arrout, arrin)
    innerdim = 0
    outerdim = 0
    for b in axes(arrin, 4)
        outerdim += 1
        innerdim = 0
        for c in axes(arrin, 3)
            for j in axes(arrin, 2)
                for i in axes(arrin, 1)
                    innerdim += 1
                    arrout[innerdim, outerdim] = arrin[i, j, c, b]
                end
            end
        end
    end
    return
end

function flattenview!(arrout, arrin)
    @views begin
        # Flatten the first 3 dimensions of `x` into `layer.a`
        for idx in axes(arrin, 4)  # iterate over batch dimension (4th dimension)
            arrout[:, idx] .= arrin[:, :, :, idx][:]  # Flatten the first 3 dimensions and assign to `layer.a`
        end
    end
end

function softmax!(a::Array{Float64,2}, z::Array{Float64,2})
    for c in axes(z, 2) # columns = samples
        va = view(a, :, c)
        vz = view(z, :, c)
        va .= exp.(vz .- maximum(vz))
        va .= va ./ (sum(va) .+ 1e-12)
    end
    return
end

function test_plot(s, e)
    plot(s:e)
end