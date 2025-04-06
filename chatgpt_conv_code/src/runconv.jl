## packages and inputs

using Convolution
using StatsBase
using BenchmarkTools

# some hyperparameters
sample_size = 10_000
minibatch_size = 50
epochs=10   # 15 epochs yields near perfect training convergence
modelspec = Convolution.two_linear
lr=0.08

## 


layerspecs, layers, x_train, y_train = preptrain(modelspec, sample_size, minibatch_size);


##

stats = train_loop!(layers; x_train=x_train, y_train=y_train, batch_size = sample_size, epochs=epochs, minibatch_size=minibatch_size, lr=lr);


##

predlayers = setup_preds(modelspec, layers, sample_size);

## 

Convolution.predict(predlayers, x_train, y_train)