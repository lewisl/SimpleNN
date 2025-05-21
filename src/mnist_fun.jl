
# requires using MLDatasets
function setup_mnist(full_batch, preptest=false)
    trainset = MNIST(:train)
    testset = MNIST(:test)
    @show size(trainset)
    @show size(testset)

    x_train = trainset.features[1:28, 1:28, 1:full_batch]
    x_train = ELT.(x_train)
    x_train = reshape(x_train, 28, 28, 1, full_batch)

    y_train = trainset.targets[1:full_batch]
    y_train = indicatormat(y_train)
    y_train = ELT.(y_train)

    preptest && begin
        x_test = testset.features[1:28, 1:28, :]
        x_test = ELT.(x_test)
        x_test = reshape(x_test, 28, 28, 1, :)

        y_test = testset.targets
        y_test = indicatormat(y_test)
        y_test = ELT.(y_test)
    end

    # shuffle the variables and outcome identically
    img_idx = shuffle(1:size(x_train, 4))
    x_train_shuf = copy(x_train[:, :, :, img_idx])
    y_train_shuf = copy(y_train[:, img_idx])

    # preptest == false
    preptest || return x_train_shuf, y_train_shuf

    # preptest == true
    preptest && return x_train_shuf, y_train_shuf, x_test, y_test
end


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