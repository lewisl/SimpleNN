# %% packages and inputs

using Convolution
# using StatsBase
# using BenchmarkTools

# %% 

# ============================
# Sample model definitions or modelspecs
# ============================


# 64 channels is not great
one_conv = LayerSpec[
    LayerSpec(h=28, w=28, outch=1, kind=:input, name=:input)
    convlayerspec(outch=32, f_h=3, f_w=3, name=:conv1, activation=:relu, adj=0.0)
    maxpoollayerspec(name=:maxpool1, f_h=2, f_w=2)
    LayerSpec(kind=:flatten, name=:flatten)
    LayerSpec(h=200, kind=:linear, activation=:relu, name=:linear1, adj=0.0)
    # LayerSpec(h=100, kind=:linear, activation=:relu, name=:linear2, adj=0.0)
    LayerSpec(h=10, kind=:linear, activation=:softmax, name=:output)
]


two_conv = LayerSpec[
    LayerSpec(name=:input, h=28, w=28, outch=1, kind=:input,)
    convlayerspec(name=:conv1, outch=24, f_h=3, f_w=3, activation=:relu, adj=0.0)
    maxpoollayerspec(name=:maxpool1, f_h=2, f_w=2)
    convlayerspec(name=:conv2, outch=48, f_h=3, f_w=3, activation=:relu, adj=0.0)
    maxpoollayerspec(name=:maxpool2, f_h=2, f_w=2)
    flattenlayerspec(name=:flatten)
    linearlayerspec(name=:linear1, output=128, adj=0.0)
    linearlayerspec(name=:linear2, output=64, adj=0.0)
    LayerSpec(name=:output, h=10, kind=:linear, activation=:softmax,)
]

two_linear = LayerSpec[
    LayerSpec(h=28, w=28, outch=1, kind=:input, name=:input)
    flattenlayerspec(name=:flatten)
    linearlayerspec(name=:linear1, output=256, adj=0.01)
    linearlayerspec(name=:linear2, output=256, adj=0.01)
    LayerSpec(h=10, kind=:linear, name=:output, activation=:softmax)
]

three_linear = LayerSpec[
    LayerSpec(h=28, w=28, outch=1, kind=:input, name=:input)
    flattenlayerspec(name=:flatten)
    linearlayerspec(name=:linear1, output=256, adj=0.01)
    linearlayerspec(name=:linear1, output=256, adj=0.01)
    linearlayerspec(name=:linear2, output=256, adj=0.01)
    LayerSpec(h=10, kind=:linear, name=:output, activation=:softmax)
]

# %%   some hyperparameters
preptest = true
@show preptest
sample_size = 10_000
minibatch_size = 50
epochs = 15   # 15 epochs yields near perfect training convergence
modelspec = two_linear
lr = 0.08

# %%

@show preptest
if !preptest
    layerspecs, layers, x_train, y_train = preptrain(modelspec, sample_size, minibatch_size, preptest=preptest)
else
    layerspecs, layers, x_train, y_train, x_test, y_test = preptrain(modelspec, sample_size, minibatch_size, preptest=preptest)
    testsize = size(y_test, 2)
end;


# %%

stats = train_loop!(layers; x_train=x_train, y_train=y_train, batch_size=sample_size, epochs=epochs, minibatch_size=minibatch_size, lr=lr);


# %%

predlayerstrain = setup_preds(modelspec, layers, sample_size);

# %%

prediction(predlayerstrain, x_train, y_train)

# %% predict with testset

predlayerstest = setup_preds(modelspec, layers, testsize);
prediction(predlayerstest, x_test, y_test)