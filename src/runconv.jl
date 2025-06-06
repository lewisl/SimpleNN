# ==========================================================
# script for running and testing different network structures
# ==========================================================

# %% startup   required for non-Julia aware environment like Zed REPL

cd(joinpath(homedir(), "code", "SimpleNN"))
using Pkg
Pkg.activate(".")
Pkg.instantiate()


# %% packages and inputs


using SimpleNN

const ELT = Float32
# %%

# ============================
# Sample model definitions (aka layerspecs)
# ============================


# 64 channels is not great
one_conv = LayerSpec[
    inputlayerspec(h=28, w=28, outch=1, name=:input)
    convlayerspec(outch=32, f_h=3, f_w=3, name=:conv1, activation=:relu, padrule=:none, optimization=:adamw, normalization=:batchnorm)  # , normalization=:batchnorm  , optimization=:adamw
    maxpoollayerspec(name=:maxpool1, f_h=2, f_w=2)
    flattenlayerspec(name=:flatten)
    linearlayerspec(outputdim=200, activation=:relu, name=:linear1,  optimization=:adamw, normalization=:batchnorm)   # , normalization=:batchnorm
    outputlayerspec(outputdim=10, activation=:softmax, name=:output)
];

le_net = LayerSpec[
    inputlayerspec(h=28, w=28, outch=1, name=:input)
    convlayerspec(outch=6, f_h=5, f_w=5, activation=:relu, name=:conv1, padrule=:none)  # optimization=:adamw, normalization=:batchnorm
    maxpoollayerspec(name=:maxpool1, f_h=2, f_w=2)
    convlayerspec(outch=16, f_h=5, f_w=5, activation=:relu, name=:conv2, padrule=:none)  # optimization=:adamw,, normalization=:batchnorm)
    maxpoollayerspec(name=:maxpool2, f_h=2, f_w=2)
    flattenlayerspec(name=:flatten)
    linearlayerspec(outputdim=120, activation=:relu, name=:linear1)   # optimization=:adamw, normalization=:batchnorm
    linearlayerspec(outputdim=84, activation=:relu, name=:linear2)   # optimization=:adamw, normalization=:batchnorm
    outputlayerspec(outputdim=10, activation=:softmax, name=:output)
]

two_conv = LayerSpec[
    inputlayerspec(name=:input, h=28, w=28, outch=1)
    convlayerspec(name=:conv1, outch=32, f_h=3, f_w=3, activation=:relu, normalization=:batchnorm, optimization=:adam)
    # maxpoollayerspec(name=:maxpool1, f_h=2, f_w=2)
    convlayerspec(name=:conv2, outch=16, f_h=3, f_w=3, activation=:relu, normalization=:batchnorm, optimization=:adam)
    maxpoollayerspec(name=:maxpool2, f_h=2, f_w=2)
    flattenlayerspec(name=:flatten)
    linearlayerspec(name=:linear1, outputdim=200, normalization=:batchnorm, optimization=:adam)
    outputlayerspec(outputdim=10, activation=:softmax, name=:output)
];

two_linear = LayerSpec[
    inputlayerspec(h=28, w=28, outch=1, name=:input)
    flattenlayerspec(name=:flatten)
    linearlayerspec(name=:linear1, outputdim=256, normalization=:batchnorm, optimization=:adam)   # normalization=:batchnorm
    linearlayerspec(name=:linear2, outputdim=256, normalization=:batchnorm, optimization=:adam)
    outputlayerspec(outputdim=10, activation=:softmax, name=:output)
];

three_linear = LayerSpec[
    inputlayerspec(h=28, w=28, outch=1, name=:input)
    flattenlayerspec(name=:flatten)
    linearlayerspec(name=:linear1, outputdim=300, normalization=:batchnorm)
    linearlayerspec(name=:linear2, outputdim=300, normalization=:batchnorm)
    linearlayerspec(name=:linear3, outputdim=300, normalization=:batchnorm)
    outputlayerspec(outputdim=10, activation=:softmax, name=:output)
];

# %%   some hyperparameters


preptest = true
full_batch = 60_000
minibatch_size = 50
epochs = 10
layerspecs = one_conv

# for le_net lr=ELT(0.0003) epochs = 5  reg=:none (until things work...)
# for one_conv lr=ELT(0.001) epochs = 10
hp = HyperParameters(lr=ELT(0.001), reg=:L2, regparm=ELT(0.00063), do_stats=false)  # reg=:L2, regparm=0.00043,

# %%  # setup the layers: set array sizes and pre-allocate data and weight arrays

layers = setup_train(layerspecs, minibatch_size);

# %%  load the data for train and testing, if applicable


if !preptest
    x_train, y_train = setup_mnist(full_batch, preptest)
else
    x_train, y_train, x_test, y_test = setup_mnist(full_batch, preptest)
    testsize = size(y_test, 2)
end;


# %%  train the model

stats = train!(layers; x=x_train, y=y_train, full_batch=full_batch,
    epochs=epochs, minibatch_size=minibatch_size, hp=hp);


# %%  predict with full training set

predlayerstrain = setup_preds(layerspecs, layers, minibatch_size);
minibatch_prediction(predlayerstrain, x_train, y_train)


# %% predict with testset

predlayerstest = setup_preds(layerspecs, layers, minibatch_size);
minibatch_prediction(predlayerstest, x_test, y_test)


# %% full batch prediction on test set, much slower  -- to verify that minibatch_prediction produces same result

predlayerstestfull = setup_preds(layerspecs, layers, testsize)
prediction(predlayerstestfull, x_test, y_test)


# %% predict a single example

samplenumber = 200;

pred1layers = setup_preds(layerspecs, layers, 1);

x_single = x_train[:, :, :, samplenumber];
x_single = reshape(x_single, 28, 28, 1, 1);

y_single = y_train[:, samplenumber];
y_single = reshape(y_single, :, 1);


SimpleNN.feedforward!(pred1layers, x_single);
pred1 = pred1layers[end].a;

target_digit = SimpleNN.find_max_idx(y_single[:, 1]) - 1;
pred_digit = SimpleNN.find_max_idx(pred1[:, 1]) - 1;

println("\nTarget digit: ", target_digit, "  Predicted digit: ", pred_digit)

# %%  # display the selected digit

display_mnist_digit(x_single)
