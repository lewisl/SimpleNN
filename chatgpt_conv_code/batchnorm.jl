function batchnorm_forward(layer::BatchNormLayer, x, training=true, momentum=0.9, eps=1e-5)
    μ_B = mean(x, dims=2)
    σ_B = std(x, dims=2) .+ eps

    x_hat = (x .- μ_B) ./ σ_B
    y = layer.γ .* x_hat .+ layer.β

    if training
        layer.running_mean .= momentum * layer.running_mean + (1 - momentum) * μ_B
        layer.running_var .= momentum * layer.running_var + (1 - momentum) * (σ_B .^ 2)
        layer.x_hat = x_hat  # Save for backprop
        layer.σ_B = σ_B
    end
    return y
end

function batchnorm_backward(layer::BatchNormLayer, d_out, x, eps=1e-5)
    m = size(x, 2)  # Batch size

    μ_B = mean(x, dims=2)
    σ_B = std(x, dims=2) .+ eps
    x_hat = (x .- μ_B) ./ σ_B

    d_gamma = sum(d_out .* x_hat, dims=2)
    d_beta = sum(d_out, dims=2)

    d_x = (1 / m) .* layer.γ ./ σ_B .* (m .* d_out .- d_beta .- x_hat .* d_gamma)

    return d_x, d_gamma, d_beta
end

function feedforward!(layers, batchnorm_layers, x_train, n_samples; training=true)
    out1 = conv_forward_unrolled!(layers.conv1, x_train, training)        
    out1_bn = batchnorm_forward(batchnorm_layers.bn1, out1, training)
    layers.conv1.relumask = out1_bn .> 0.0
    relu_conv1 = out1_bn .* layers.conv1.relumask

    out2 = conv_forward_unrolled!(layers.conv2, relu_conv1, training)     
    out2_bn = batchnorm_forward(batchnorm_layers.bn2, out2, training)
    layers.conv2.relumask = out2_bn .> 0.0
    relu_conv2 = out2_bn .* layers.conv2.relumask

    flat = reshape(out2_bn, :, n_samples)

    lin1_out_vec = linear_forward(layers.lin1, flat)
    lin1_bn = batchnorm_forward(batchnorm_layers.bn3, lin1_out_vec, training)
    layers.lin1.relumask = lin1_bn .> 0.0
    relu1 = lin1_bn .* layers.lin1.relumask

    lin2_out_vec = linear_forward(layers.lin2, relu1, training)
    lin2_bn = batchnorm_forward(batchnorm_layers.bn4, lin2_out_vec, training)
    layers.lin2.relumask = lin2_bn .> 0.0
    relu2 = lin2_bn .* layers.lin2.relumask

    lin3_out_vec = linear_forward(layers.lin3, relu2, training)
    lin3_bn = batchnorm_forward(batchnorm_layers.bn5, lin3_out_vec, training)
    layers.lin3.relumask = lin3_bn .> 0.0
    relu3 = lin3_bn .* layers.lin3.relumask

    lin4_out_vec = linear_forward(layers.lin4, relu3, training)
    lin4_bn = batchnorm_forward(batchnorm_layers.bn6, lin4_out_vec, training)
    layers.lin4.relumask = lin4_bn .> 0.0
    relu4 = lin4_bn .* layers.lin4.relumask

    lin5_out_vec = linear_forward(layers.lin5, relu4, training)

    probs = softmax(lin5_out_vec)
end

function backprop!(layers, batchnorm_layers, y_train, probs, n_samples, lr)
    dlin5 = dloss_dz(probs, y_train)

    drelu4 = linear_backward(layers.lin5, dlin5, n_samples)
    dlin4_out = drelu4 .* layers.lin4.relumask

    # BatchNorm Backprop
    dlin4_out_bn, d_gamma6, d_beta6 = batchnorm_backward(batchnorm_layers.bn6, dlin4_out, layers.lin4.x)
    batchnorm_layers.bn6.γ .-= lr .* d_gamma6
    batchnorm_layers.bn6.β .-= lr .* d_beta6

    drelu3 = linear_backward(layers.lin4, dlin4_out_bn, n_samples)
    dlin3_out = drelu3 .* layers.lin3.relumask
    dlin3_out_bn, d_gamma5, d_beta5 = batchnorm_backward(batchnorm_layers.bn5, dlin3_out, layers.lin3.x)
    batchnorm_layers.bn5.γ .-= lr .* d_gamma5
    batchnorm_layers.bn5.β .-= lr .* d_beta5

    drelu2 = linear_backward(layers.lin3, dlin3_out_bn, n_samples)
    dlin2_out = drelu2 .* layers.lin2.relumask
    dlin2_out_bn, d_gamma4, d_beta4 = batchnorm_backward(batchnorm_layers.bn4, dlin2_out, layers.lin2.x)
    batchnorm_layers.bn4.γ .-= lr .* d_gamma4
    batchnorm_layers.bn4.β .-= lr .* d_beta4

    drelu1 = linear_backward(layers.lin2, dlin2_out_bn, n_samples)
    dlin1_out = drelu1 .* layers.lin1.relumask
    dlin1_out_bn, d_gamma3, d_beta3 = batchnorm_backward(batchnorm_layers.bn3, dlin1_out, layers.lin1.x)
    batchnorm_layers.bn3.γ .-= lr .* d_gamma3
    batchnorm_layers.bn3.β .-= lr .* d_beta3

    dflat = linear_backward(layers.lin1, dlin1_out_bn, n_samples)
    dimg = reshape(dflat, size(layers.conv2.relumask))
    drelu_conv2_out = dimg
    dconv2_out = drelu_conv2_out .* layers.conv2.relumask

    drelu_conv1_out = conv_backward_unrolled!(layers.conv2, dconv2_out)
    dconv1_out = drelu_conv1_out .* layers.conv1.relumask
    _ = conv_backward_unrolled!(layers.conv1, dconv1_out)

    # Update weights
    update!(layers.conv1.weight, layers.conv1.grad_weight, lr)
    update!(layers.conv1.bias, layers.conv1.grad_bias, lr)

    update!(layers.conv2.weight, layers.conv2.grad_weight, lr)
    update!(layers.conv2.bias, layers.conv2.grad_bias, lr)

    update!(layers.lin1.weight, layers.lin1.grad_weight, lr)
    update!(layers.lin1.bias, layers.lin1.grad_bias, lr)

    update!(layers.lin2.weight, layers.lin2.grad_weight, lr)
    update!(layers.lin2.bias, layers.lin2.grad_bias, lr)

    update!(layers.lin3.weight, layers.lin3.grad_weight, lr)
    update!(layers.lin3.bias, layers.lin3.grad_bias, lr)

    update!(layers.lin4.weight, layers.lin4.grad_weight, lr)
    update!(layers.lin4.bias, layers.lin4.grad_bias, lr)

    update!(layers.lin5.weight, layers.lin5.grad_weight, lr)
    update!(layers.lin5.bias, layers.lin5.grad_bias, lr)
end
