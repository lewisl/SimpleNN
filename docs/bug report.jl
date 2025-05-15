### Model output for abstract function field type:
```plaintext
name: input
arrays:
a: (28, 28, 1, 50)


name: conv1
arrays:
z: (28, 28, 32, 50)
z_norm: (28, 28, 32, 50)
a: (28, 28, 32, 50)
a_below: (28, 28, 1, 50)
pad_a_below: (30, 30, 1, 50)
eps_l: (28, 28, 1, 50)
pad_next_eps: (28, 28, 32, 50)
grad_a: (28, 28, 32, 50)
pad_x: (30, 30, 1, 50)
weight: (3, 3, 1, 32)
bias: (32,)
grad_weight: (3, 3, 1, 32)
grad_bias: (32,)
functions:
activationf: relu! typeof(SimpleNN.relu!)
activation_gradf: relu_grad! typeof(SimpleNN.relu_grad!)
normalizationf: batchnorm! typeof(SimpleNN.batchnorm!)
normalization_gradf: batchnorm_grad! typeof(SimpleNN.batchnorm_grad!)


name: maxpool1
arrays:
a: (14, 14, 32, 50)
mask: (28, 28, 32, 50)
eps_l: (28, 28, 32, 50)


name: flatten
arrays:
dl_dflat: (6272, 50)
a: (6272, 50)
eps_l: (14, 14, 32, 50)


name: linear1
arrays:
z: (200, 50)
z_norm: (200, 50)
a: (200, 50)
grad_a: (200, 50)
a_below: (6272, 50)
eps_l: (200, 50)
weight: (200, 6272)
bias: (200,)
grad_weight: (200, 6272)
grad_bias: (200,)
functions:
activationf: relu! typeof(SimpleNN.relu!)
activation_gradf: relu_grad! typeof(SimpleNN.relu_grad!)
normalizationf: batchnorm! typeof(SimpleNN.batchnorm!)
normalization_gradf: batchnorm_grad! typeof(SimpleNN.batchnorm_grad!)


name: output
arrays:
z: (10, 50)
z_norm: (0, 0)
a: (10, 50)
grad_a: (10, 50)
a_below: (200, 50)
eps_l: (10, 50)
weight: (10, 200)
bias: (10,)
grad_weight: (10, 200)
grad_bias: (10,)
functions:
activationf: softmax! typeof(SimpleNN.softmax!)
activation_gradf: noop typeof(SimpleNN.noop)
normalizationf: noop typeof(SimpleNN.noop)
normalization_gradf: noop typeof(SimpleNN.noop)
```

### model output for parametric types for function fields:
```plaintext
name: input
arrays:
a: (28, 28, 1, 50)


name: conv1
arrays:
z: (28, 28, 32, 50)
z_norm: (28, 28, 32, 50)
a: (28, 28, 32, 50)
a_below: (28, 28, 1, 50)
pad_a_below: (30, 30, 1, 50)
eps_l: (28, 28, 1, 50)
pad_next_eps: (28, 28, 32, 50)
grad_a: (28, 28, 32, 50)
pad_x: (30, 30, 1, 50)
weight: (3, 3, 1, 32)
bias: (32,)
grad_weight: (3, 3, 1, 32)
grad_bias: (32,)
functions:
activationf: relu! typeof(SimpleNN.relu!)
activation_gradf: relu_grad! typeof(SimpleNN.relu_grad!)
normalizationf: batchnorm! typeof(SimpleNN.batchnorm!)
normalization_gradf: batchnorm_grad! typeof(SimpleNN.batchnorm_grad!)


name: maxpool1
arrays:
a: (14, 14, 32, 50)
mask: (28, 28, 32, 50)
eps_l: (28, 28, 32, 50)


name: flatten
arrays:
dl_dflat: (6272, 50)
a: (6272, 50)
eps_l: (14, 14, 32, 50)


name: linear1
arrays:
z: (200, 50)
z_norm: (200, 50)
a: (200, 50)
grad_a: (200, 50)
a_below: (6272, 50)
eps_l: (200, 50)
weight: (200, 6272)
bias: (200,)
grad_weight: (200, 6272)
grad_bias: (200,)
functions:
activationf: relu! typeof(SimpleNN.relu!)
activation_gradf: relu_grad! typeof(SimpleNN.relu_grad!)
normalizationf: batchnorm! typeof(SimpleNN.batchnorm!)
normalization_gradf: batchnorm_grad! typeof(SimpleNN.batchnorm_grad!)


name: output
arrays:
z: (10, 50)
z_norm: (0, 0)
a: (10, 50)
grad_a: (10, 50)
a_below: (200, 50)
eps_l: (10, 50)
weight: (10, 200)
bias: (10,)
grad_weight: (10, 200)
grad_bias: (10,)
functions:
activationf: softmax! typeof(SimpleNN.softmax!)
activation_gradf: noop typeof(SimpleNN.noop)
normalizationf: noop typeof(SimpleNN.noop)
normalization_gradf: noop typeof(SimpleNN.noop)
```

### wrong results from parametric type code
```julia
predlayerstrain = setup_preds(layerspecs, layers, minibatch_size);
       minibatch_prediction(predlayerstrain, x_train, y_train)
(0.06931666666666667, 8.096017504918363)
```