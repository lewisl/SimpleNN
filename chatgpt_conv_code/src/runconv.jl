## packages and inputs

using Convolution
using StatsBase
using BenchmarkTools

# some hyperparameters
preptest = true
@show preptest
sample_size = 50_000
minibatch_size = 50
epochs=24   # 15 epochs yields near perfect training convergence
modelspec = Convolution.one_conv
lr=0.1

## 

@show preptest
if !preptest
    layerspecs, layers, x_train, y_train = preptrain(modelspec, sample_size, minibatch_size, preptest=preptest);
else
    layerspecs, layers, x_train, y_train, x_test, y_test = preptrain(modelspec, sample_size, minibatch_size, preptest=preptest);
    testsize = size(y_test, 2)
end;


##

stats = train_loop!(layers; x_train=x_train, y_train=y_train, batch_size = sample_size, epochs=epochs, minibatch_size=minibatch_size, lr=lr);


##

predlayerstrain = setup_preds(modelspec, layers, sample_size);

## 

Convolution.predict(predlayerstrain, x_train, y_train)

## predict with testset

predlayerstest = setup_preds(modelspec, layers, testsize);
Convolution.predict(predlayerstest, x_test, y_test)