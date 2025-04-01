## packages and inputs

using Convolution
using StatsBase


sample_size = 10_000
minibatch_size = 50
epochs=15    # 20 epochs yields near perfect training convergence
lr=0.06

## 

layerspecs, layers, x_train, y_train = preptrain(Convolution.small_conv, sample_size, minibatch_size);


##

stats = train_loop!(layers; x_train=x_train, y_train=y_train, batch_size = sample_size, epochs=epochs, minibatch_size=minibatch_size, lr=lr);

##