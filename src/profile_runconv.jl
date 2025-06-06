# %% packages and inputs


using SimpleNN
using Profile
using BenchmarkTools


# %%

# ============================
# Sample model definitions (aka layerspecs)
# ============================


# 64 channels is not great
one_conv = LayerSpec[
    LayerSpec(h=28, w=28, outch=1, kind=:input, name=:input)
    convlayerspec(outch=32, f_h=3, f_w=3, name=:conv1, activation=:relu, normalization=:batchnorm)  # , normalization=:batchnorm  , optimization=:adamw
    maxpoollayerspec(name=:maxpool1, f_h=2, f_w=2)
    flattenlayerspec(name=:flatten)
    linearlayerspec(output=200, activation=:relu, name=:linear1, normalization=:batchnorm)
    LayerSpec(h=10, kind=:linear, name=:output, activation=:softmax)
];


two_conv = LayerSpec[
    LayerSpec(name=:input, h=28, w=28, outch=1, kind=:input)
    convlayerspec(name=:conv1, outch=24, f_h=3, f_w=3, activation=:relu)
    maxpoollayerspec(name=:maxpool1, f_h=2, f_w=2)
    convlayerspec(name=:conv2, outch=48, f_h=3, f_w=3, activation=:relu)
    maxpoollayerspec(name=:maxpool2, f_h=2, f_w=2)
    flattenlayerspec(name=:flatten)
    linearlayerspec(name=:linear1, output=200)
    LayerSpec(name=:output, h=10, kind=:linear, activation=:softmax)
];

two_linear = LayerSpec[
    LayerSpec(h=28, w=28, outch=1, kind=:input, name=:input)
    flattenlayerspec(name=:flatten)
    linearlayerspec(name=:linear1, output=256)   # normalization=:batchnorm
    linearlayerspec(name=:linear2, output=256)
    LayerSpec(h=10, kind=:linear, name=:output, activation=:softmax)
];

three_linear = LayerSpec[
    LayerSpec(h=28, w=28, outch=1, kind=:input, name=:input)
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
epochs = 1  # 15 epochs yields near perfect training convergence
layerspecs = one_conv

hp = HyperParameters(lr=ELT(0.05), reg=:none, regparm=ELT(0.00043), do_stats=false)  # reg=:L2, regparm=0.00043,

# %%

layers = setup_train(layerspecs, minibatch_size);

if !preptest
    x_train, y_train = setup_mnist(full_batch, preptest);
else
    x_train, y_train, x_test, y_test = setup_mnist(full_batch, preptest);
    testsize = size(y_test, 2);
end;


# %%

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

samplenumber = 20;

pred1layers = setup_preds(layerspecs, layers, 1);

x_single = x_train[:, :, :, samplenumber];
x_single = reshape(x_single, 28, 28, 1, 1);

y_single = y_train[:, samplenumber];
y_single = reshape(y_single, :, 1);


display_mnist_digit(x_single)

SimpleNN.feedforward!(pred1layers, x_single, 1);
pred1 = pred1layers[end].a;

target_digit = SimpleNN.find_max_idx(y_single[:, 1]) - 1;
pred_digit = SimpleNN.find_max_idx(pred1[:, 1]) - 1;

println("\nTarget digit: ", target_digit, "  Predicted digit: ", pred_digit)

# --- Profiling the training loop ---
Profile.clear()
@profile begin
    stats = train!(layers; x=x_train, y=y_train, full_batch=full_batch, epochs=epochs, minibatch_size=minibatch_size, hp=hp)
end
Profile.print()

# --- Micro-benchmarks on a representative layer (e.g., first ConvLayer) ---
rep_layer = layers[2]  # adjust index if needed for your model

println("\nBenchmarking update_weights! on rep_layer:")
@btime update_weights!($rep_layer, $hp, 1)

println("\nBenchmarking update_batchnorm! on rep_layer:")
@btime update_batchnorm!($rep_layer, $hp, 1)

println("\nBenchmarking pre_adam! on rep_layer:")
@btime pre_adam!($rep_layer, $rep_layer.optparams, 1)

println("\nBenchmarking pre_adam_batchnorm! on rep_layer:")
@btime pre_adam_batchnorm!($rep_layer.normparams, $rep_layer.optparams, 1)

println("\nBenchmarking adam_helper! on rep_layer weights:")
@btime adam_helper!($rep_layer.grad_m_weight, $rep_layer.grad_v_weight, $rep_layer.grad_weight, $rep_layer.optparams, 1)
