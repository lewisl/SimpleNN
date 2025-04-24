# %% packages and inputs

using Convolution


# %% 

# ============================
# Sample model definitions (aka layerspecs)
# ============================


# 64 channels is not great
one_conv = LayerSpec[
    LayerSpec(h=28, w=28, outch=1, kind=:input, name=:input)
    convlayerspec(outch=32, f_h=3, f_w=3, name=:conv1, activation=:relu)
    maxpoollayerspec(name=:maxpool1, f_h=2, f_w=2)
    LayerSpec(kind=:flatten, name=:flatten)
    LayerSpec(h=200, kind=:linear, activation=:relu, name=:linear1)
    # LayerSpec(h=100, kind=:linear, activation=:relu, name=:linear2, adj=0.0)
    LayerSpec(h=10, kind=:linear, activation=:softmax, name=:output)
];


two_conv = LayerSpec[
    LayerSpec(name=:input, h=28, w=28, outch=1, kind=:input,)
    convlayerspec(name=:conv1, outch=24, f_h=3, f_w=3, activation=:relu)
    maxpoollayerspec(name=:maxpool1, f_h=2, f_w=2)
    convlayerspec(name=:conv2, outch=48, f_h=3, f_w=3, activation=:relu)
    maxpoollayerspec(name=:maxpool2, f_h=2, f_w=2)
    flattenlayerspec(name=:flatten)
    linearlayerspec(name=:linear1, output=200)
    LayerSpec(name=:output, h=10, kind=:linear, activation=:softmax,)
];

two_linear = LayerSpec[
    LayerSpec(h=28, w=28, outch=1, kind=:input, name=:input)
    flattenlayerspec(name=:flatten)
    linearlayerspec(name=:linear1, output=256)
    linearlayerspec(name=:linear2, output=256)
    LayerSpec(h=10, kind=:linear, name=:output, activation=:softmax)
];

three_linear = LayerSpec[
    LayerSpec(h=28, w=28, outch=1, kind=:input, name=:input)
    flattenlayerspec(name=:flatten)
    linearlayerspec(name=:linear1, output=256, adj=0.0)
    linearlayerspec(name=:linear1, output=256, adj=0.0)
    linearlayerspec(name=:linear2, output=256, adj=0.0)
    LayerSpec(h=10, kind=:linear, name=:output, activation=:softmax)
];

# %%   some hyperparameters
preptest = true
full_batch = 60_000
minibatch_size = 50
epochs = 20   # 15 epochs yields near perfect training convergence
layerspecs = two_linear

hp = HyperParameters(lr=0.04, reg=:L2, regparm=0.0006, do_stats=false)  # reg=:L2, regparm=0.002

# %%

layers = setup_train(layerspecs, minibatch_size)

if !preptest
    x_train, y_train = setup_mnist(full_batch, preptest)
else
    x_train, y_train, x_test, y_test = setup_mnist(full_batch, preptest)
    testsize = size(y_test, 2)
end;


# %%

stats = train_loop!(layers; x=x_train, y=y_train, full_batch=full_batch, 
        epochs=epochs, minibatch_size=minibatch_size, hp=hp);


# %%  predict with full training set

predlayerstrain = setup_preds(layerspecs, layers, full_batch);
prediction(predlayerstrain, x_train, y_train)

# %% predict with testset

predlayerstest = setup_preds(layerspecs, layers, testsize);
prediction(predlayerstest, x_test, y_test)

# %% predict a single example

samplenumber = 20;

pred1layers = Convolution.setup_preds(layerspecs, layers, 1);

x_single = x_train[:,:,:,samplenumber];
x_single = reshape(x_single,28,28,1,1);

y_single = y_train[:,samplenumber];
y_single = reshape(y_single,:,1);


display_mnist_digit(x_single)

Convolution.feedforward!(pred1layers, x_single, 1);
pred1 = pred1layers[end].a;

target_digit = Convolution.find_max_idx(y_single[:,1])-1;
pred_digit =  Convolution.find_max_idx(pred1[:,1 ])-1;

println("\nTarget digit: ", target_digit, "  Predicted digit: ", pred_digit)
