# %% packages and inputs


using SimpleNN


# %%

# ============================
# Sample model definitions (aka layerspecs)
# ============================


# 64 channels is not great
one_conv = LayerSpec[
    inputlayerspec(h=28, w=28, outch=1, name=:input)
    convlayerspec(outch=32, f_h=3, f_w=3, name=:conv1, activation=:relu, normalization=:batchnorm, optimization=:adam)  # , normalization=:batchnorm  , optimization=:adamw
    maxpoollayerspec(name=:maxpool1, f_h=2, f_w=2)
    flattenlayerspec(name=:flatten)
    linearlayerspec(output=200, activation=:relu, name=:linear1, normalization=:batchnorm, optimization=:adam)
    LayerSpec(h=10, kind=:linear, name=:output, activation=:softmax)
];


two_conv = LayerSpec[
    inputlayerspec(name=:input, h=28, w=28, outch=1)
    convlayerspec(name=:conv1, outch=32, f_h=3, f_w=3, activation=:relu,normalization=:batchnorm, optimization=:adam)
    # maxpoollayerspec(name=:maxpool1, f_h=2, f_w=2)
    convlayerspec(name=:conv2, outch=16, f_h=3, f_w=3, activation=:relu,normalization=:batchnorm, optimization=:adam)
    maxpoollayerspec(name=:maxpool2, f_h=2, f_w=2)
    flattenlayerspec(name=:flatten)
    linearlayerspec(name=:linear1, output=200,normalization=:batchnorm, optimization=:adam)
    LayerSpec(name=:output, h=10, kind=:linear, activation=:softmax)
];

two_linear = LayerSpec[
    inputlayerspec(h=28, w=28, outch=1, name=:input)
    flattenlayerspec(name=:flatten)
    linearlayerspec(name=:linear1, output=256, normalization=:batchnorm,  optimization=:adam)   # normalization=:batchnorm
    linearlayerspec(name=:linear2, output=256, normalization=:batchnorm,  optimization=:adam)
    LayerSpec(h=10, kind=:linear, name=:output, activation=:softmax)
];

three_linear = LayerSpec[
    inputlayerspec(h=28, w=28, outch=1,  name=:input)
    flattenlayerspec(name=:flatten)
    linearlayerspec(name=:linear1, output=300, normalization=:batchnorm)
    linearlayerspec(name=:linear2, output=300, normalization=:batchnorm)
    linearlayerspec(name=:linear3, output=300, normalization=:batchnorm)
    LayerSpec(h=10, kind=:linear, name=:output, activation=:softmax)
];

# %%   some hyperparameters
preptest = true
full_batch = 60_000
minibatch_size = 50
epochs = 5  # 15 epochs yields near perfect training convergence with dense linear layers
layerspecs = one_conv

hp = HyperParameters(lr=0.0005, reg=:L2, regparm=0.00043, do_stats=false)  # reg=:L2, regparm=0.00043,

# %%

layers = setup_train(layerspecs, minibatch_size);

if !preptest
    x_train, y_train = setup_mnist(full_batch, preptest);
else
    x_train, y_train, x_test, y_test = setup_mnist(full_batch, preptest);
    testsize = size(y_test, 2);
end;


# %%

stats = train_loop!(layers; x=x_train, y=y_train, full_batch=full_batch,
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

samplenumber = 20;

pred1layers = setup_preds(layerspecs, layers, 1);

x_single = x_train[:, :, :, samplenumber];
x_single = reshape(x_single, 28, 28, 1, 1);

y_single = y_train[:, samplenumber];
y_single = reshape(y_single, :, 1);


display_mnist_digit(x_single)

Convolution.feedforward!(pred1layers, x_single, 1);
pred1 = pred1layers[end].a;

target_digit = Convolution.find_max_idx(y_single[:, 1]) - 1;
pred_digit = Convolution.find_max_idx(pred1[:, 1]) - 1;

println("\nTarget digit: ", target_digit, "  Predicted digit: ", pred_digit)
