# Some Results

#### one_conv 10 epochs
Model:
```julia
one_conv = LayerSpec[
    LayerSpec(h=28, w=28, outch=1, kind=:input, name=:input)
    convlayerspec(outch=32, f_h=3, f_w=3, name=:conv1, activation=:relu, adj=0.0)
    maxpoollayerspec(name=:maxpool1, f_h=2, f_w=2)
    LayerSpec(kind=:flatten, name=:flatten)
    LayerSpec(h=200, kind=:linear, activation=:relu, name=:linear1, adj=0.0)
    # LayerSpec(h=100, kind=:linear, activation=:relu, name=:linear2, adj=0.0)
    LayerSpec(h=10, kind=:linear, activation=:softmax, name=:output)
];
```
Hyperparameters:
```julia
preptest = true
full_batch = 60_000
minibatch_size = 50
epochs = 10   # 15 epochs yields near perfect training convergence
layerspecs = one_conv

hp = HyperParameters(lr=0.09, reg=:L2, regparm=0.0006)
```

Results:
> Training: Accuracy 0.9883833333333333  Cost 0.08349290239386865
> Test: Accuracy 0.9795  Cost 0.11776864422331167

#### one_conv 15 epochs
Model:
```julia
one_conv = LayerSpec[
    LayerSpec(h=28, w=28, outch=1, kind=:input, name=:input)
    convlayerspec(outch=32, f_h=3, f_w=3, name=:conv1, activation=:relu, adj=0.0)
    maxpoollayerspec(name=:maxpool1, f_h=2, f_w=2)
    LayerSpec(kind=:flatten, name=:flatten)
    LayerSpec(h=200, kind=:linear, activation=:relu, name=:linear1, adj=0.0)
    # LayerSpec(h=100, kind=:linear, activation=:relu, name=:linear2, adj=0.0)
    LayerSpec(h=10, kind=:linear, activation=:softmax, name=:output)
];
```
Hyperparameters:
```julia
preptest = true
full_batch = 60_000
minibatch_size = 50
epochs = 15   # 15 epochs yields near perfect training convergence
layerspecs = one_conv

hp = HyperParameters(lr=0.09, reg=:L2, regparm=0.0006)
```

Results:
> Training: Accuracy 0.9893666666666666  Cost 0.08542955219525523
> Test: Accuracy 0.9809  Cost 0.1166747099312071

#### one_conv 10 epochs without batch normalization for the conv layer

Model:
```julia
one_conv = LayerSpec[
    LayerSpec(h=28, w=28, outch=1, kind=:input, name=:input)
    convlayerspec(outch=32, f_h=3, f_w=3, name=:conv1, activation=:relu, )  #normalization=:batchnorm
    maxpoollayerspec(name=:maxpool1, f_h=2, f_w=2)
    flattenlayerspec(name=:flatten)
    linearlayerspec(output=200, activation=:relu, name=:linear1, normalization=:batchnorm)
    LayerSpec(h=10, kind=:linear, name=:output, activation=:softmax)
];
```

Hyperparameters:
```julia
preptest = true
full_batch = 60_000
minibatch_size = 50
epochs = 10   # 15 epochs yields near perfect training convergence
layerspecs = one_conv

hp = HyperParameters(lr=0.1, reg=:L2, regparm=0.0004, do_stats=false)
```

Results:
> Training: Accuracy 0.9970666666666667  Cost 0.028493152668834804
> 
> Test: Accuracy 0.9836  Cost 0.09114865168921367

#### two_conv 10 epochs

Model:
```julia
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
```

Hyperparameters:

```julia
preptest = true
full_batch = 60_000
minibatch_size = 50
epochs = 10   # 15 epochs yields near perfect training convergence
layerspecs = two_conv

hp = HyperParameters(lr=0.004, reg=:L2, regparm=0.0006, do_stats=false)
```

Results:
> Training: Accuracy 0.9785666666666667  Cost 0.12997403008866698
> Test: Accuracy 0.9714  Cost 0.1598375700439552


#### three_linear 20 epochs

Model:
```julia
three_linear = LayerSpec[
    LayerSpec(h=28, w=28, outch=1, kind=:input, name=:input)
    flattenlayerspec(name=:flatten)
    linearlayerspec(name=:linear1, output=256)
    linearlayerspec(name=:linear1, output=256)
    linearlayerspec(name=:linear2, output=256)
    LayerSpec(h=10, kind=:linear, name=:output, activation=:softmax)
];
```

Hyperparameters:

```julia
preptest = true
full_batch = 60_000
minibatch_size = 50
epochs = 20   # 15 epochs yields near perfect training convergence
layerspecs = three_linear

hp = HyperParameters(lr=0.1, reg=:L2, regparm=0.0005, do_stats=false)
```

Results:
> Training: Accuracy 0.9982  Cost 0.02208198240164045
> Test: Accuracy 0.9815  Cost 0.10239645963369148

#### one_conv 10 epochs with batch normalization

Model:
```julia
one_conv = LayerSpec[
    LayerSpec(h=28, w=28, outch=1, kind=:input, name=:input)
    convlayerspec(outch=32, f_h=3, f_w=3, name=:conv1, activation=:relu, normalization=:batchnorm)  # normalization=:batchnorm
    maxpoollayerspec(name=:maxpool1, f_h=2, f_w=2)
    flattenlayerspec(name=:flatten)
    linearlayerspec(output=200, activation=:relu, name=:linear1, normalization=:batchnorm)
    LayerSpec(h=10, kind=:linear, name=:output, activation=:softmax)
];
```

Hyperparameters:
```julia
preptest = true
full_batch = 60_000
minibatch_size = 50
epochs = 10  # 15 epochs yields near perfect training convergence
layerspecs = one_conv

hp = HyperParameters(lr=0.1, reg=:L2, regparm=0.0004, do_stats=false)
```

Results:
> Training:
> Test: Accuracy 0.9854  Cost 0.08455229387921524

#### three_linear 20 epochs with batch normalization
Model:
```julia
three_linear = LayerSpec[
    LayerSpec(h=28, w=28, outch=1, kind=:input, name=:input)
    flattenlayerspec(name=:flatten)
    linearlayerspec(name=:linear1, output=300, normalization=:batchnorm)
    linearlayerspec(name=:linear2, output=300, normalization=:batchnorm)
    linearlayerspec(name=:linear3, output=300, normalization=:batchnorm)
    LayerSpec(h=10, kind=:linear, name=:output, activation=:softmax)
];
```

Hyperparameters
```julia
preptest = true
full_batch = 60_000
minibatch_size = 50
epochs = 20  # 15 epochs yields near perfect training convergence
layerspecs = three_linear

hp = HyperParameters(lr=0.1, reg=:L2, regparm=0.00043, do_stats=false)
```

Results:
> Training: Accuracy 0.9993166666666666 Cost 0.005766895445629323
> Test: 0.9817 Cost 0.11920180572947686
