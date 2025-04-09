using StatsBase
includet("sample_code.jl")
using MLDatasets


function random_onehot(i,j)
    arr = zeros(i,j)
    for n in axes(arr,2)
        rowselector = rand(1:10)
        arr[rowselector, n] = 1.0
    end
    return arr
end

function preptest(n_samples)
    layerspecs = set_layer_specs()

    display(layerspecs); println()

    layers = allocate_layers(layerspecs, x_train=rand(28,28,1,n_samples);
            y_train=rand(10,n_samples), n_samples=n_samples)

    return layerspecs, layers
end

#TODO  not a good test because there is no data input
function runtest(layerspecs, layers, n_samples)
    newfeedforward!(layers, rand(28, 28, 1, n_samples))

    newbackprop!(layers, y_train=random_onehot(10, n_samples))

    update_weight_loop!(layers)
end


function preptrain(modelspecs::Function, batch_size, mini_batch_size)
    trainset = MNIST(:train)
    testset = MNIST(:test)

    # batch_size = 10000
    # minibatch_size = 50
    x_train = trainset.features[1:28, 1:28,1:batch_size]
    x_train = Float64.(x_train)
    x_train = reshape(x_train, 28, 28, 1, batch_size)
    @show size(x_train)

    y_train = trainset.targets[1:batch_size]
    y_train = indicatormat(y_train)
    y_train = Float64.(y_train)

    # shuffle the variables and outcome identically
    img_idx = shuffle(1:size(x_train,4))
    x_train_shuf = x_train[:,:,:,img_idx]
    y_train_shuf = y_train[:,img_idx]

    layerspecs = modelspecs()   # set_layer_specs()
    display(layerspecs);println()

    layers = allocate_layers(layerspecs, mini_batch_size);

    return layerspecs, layers, x_train_shuf, y_train_shuf
end