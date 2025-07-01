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
# Sample model definitions 
# ============================

linear_reg1 = LayerSpec[
    inputlayerspec(outputdim=1, name=:input)
    outputlayerspec(outputdim=1, activation=:regression, name=:output)
];


# linear_wrong = LayerSpec[
#     inputlayerspec(outputdim=1, name=:input)
#     linearlayerspec(outputdim=1, name=:linear1)
#     outputlayerspec(outputdim=1, activation=:regression, name=:output)
# ];

# %%   some hyperparameters


preptest = false
full_batch = 5000
minibatch_size = 5000
epochs = 500
layerspecs = linear_reg1


hp = HyperParameters(lr=ELT(0.001), reg=:L2, regparm=ELT(0.00063), do_stats=false)  # reg=:L2, regparm=0.00043,

# %%  # setup the layers: set array sizes and pre-allocate data and weight arrays

layers = setup_train(layerspecs, minibatch_size);

# %%  load the data for train and testing, if applicable


# for linear regression we just need to load the training x and y

xspec = [(3.0, 0.1)]
slope  = [1.5]
b = 0.4  # intercept

x_train, y_train = lr_data(xspec, full_batch, slope, b);
x_train = Float32.(x_train);
y_train = Float32.(y_train);

    # if !preptest
    #     x_train, y_train = setup_mnist(full_batch, preptest)
    # else
    #     x_train, y_train, x_test, y_test = setup_mnist(full_batch, preptest)
    #     testsize = size(y_test, 2)
    # end;


# %%  train the model

stats = train!(layers; x=x_train, y=y_train, full_batch=full_batch,
    epochs=epochs, minibatch_size=minibatch_size, hp=hp);


# %%  predict with full training set

predlayerstrain = setup_preds(layerspecs, layers, minibatch_size);
minibatch_prediction(predlayerstrain, x_train, y_train)


# %% predict with testset

predlayerstest = setup_preds(layerspecs, layers, minibatch_size);
minibatch_prediction(predlayerstest, x_test, y_test, mse_cost)


# %% full batch prediction on test set, much slower  -- to verify that minibatch_prediction produces same result

predlayerstestfull = setup_preds(layerspecs, layers, testsize)
prediction(predlayerstestfull, x_test, y_test)


# %% predict a single example
