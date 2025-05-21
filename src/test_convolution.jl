using LinearAlgebra

# Tiny 5x5x1 single image input
x_tiny = reshape([1.0, 2, 3, 4, 5,
                  6, 7, 8, 9, 10,
                  11, 12, 13, 14, 15,
                  16, 17, 18, 19, 20,
                  21, 22, 23, 24, 25], 5,5,1,1)

# Tiny 3x3x1 single filter
w_tiny = reshape([1.0, 0, -1,
                  2, 0, -2,
                  1, 0, -1], 3,3,1,1)

# Bias (1 filter)
b_tiny = [0.0]

# Expected output size (3x3x1) due to valid convolution
out_size = (3,3,1,1)

# Create layer with manually set weights
conv_layer = ConvLayer(w_tiny, b_tiny, falses(0,0,0,0), zeros(ELT, 0,0,0,0), zeros(size(w_tiny)), zeros(size(b_tiny)))

# Perform forward pass
z_out = conv_forward_unrolled!(conv_layer, x_tiny, training=true)
println("Forward Output:\n", z_out[:,:,1,1])

# Simulated gradient coming from next layer (same size as output)
d_out = reshape([1.0, -1, 2,
                 0,  3, -2,
                 -1, 0, 1], 3,3,1,1)


dx_tiny = conv_backward_unrolled!(conv_layer, d_out)
println("Gradient w.r.t. Input (dx):\n", dx_tiny[:,:,1,1])
println("Gradient w.r.t. Weights (dw):\n", conv_layer.grad_weight[:,:,1,1])
println("Gradient w.r.t. Bias (db):\n", conv_layer.grad_bias)


function numerical_gradient_wrt_w(conv_layer, x_tiny, d_out, epsilon=1e-5)
    numerical_grad_w = zeros(ELT, size(conv_layer.weight))
    for i in eachindex(conv_layer.weight)
        conv_layer.weight[i] += epsilon
        out_plus = conv_forward_unrolled!(conv_layer, x_tiny, training=false)

        conv_layer.weight[i] -= 2epsilon
        out_minus = conv_forward_unrolled!(conv_layer, x_tiny, training=false)

        conv_layer.weight[i] += epsilon # Restore original weight

        numerical_grad_w[i] = sum((out_plus .- out_minus) .* d_out) / (2 * epsilon)
    end
    return numerical_grad_w
end

# Compute numerical gradients
num_grad_w = numerical_gradient_wrt_w(conv_layer, x_tiny, d_out)
println("Numerical Gradient w.r.t. Weights:\n", num_grad_w[:,:,1,1])

# Compare to computed gradient
println("Difference:\n", abs.(num_grad_w - conv_layer.grad_weight[:,:,1,1]))
