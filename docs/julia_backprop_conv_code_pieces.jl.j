using LinearAlgebra

# Define activation functions and their derivatives
function relu(x)
    return max.(0, x)
end

function relu_prime(x)
    return x .> 0
end

function softmax(x)
    exp_x = exp.(x .- maximum(x))
    return exp_x ./ sum(exp_x)
end

function cross_entropy_loss(pred, true)
    return -sum(true .* log.(pred))
end

# Forward pass
function forward(x, w1, w2, w3, w4)
    z1 = conv2d(x, w1)
    a1 = relu(z1)
    
    z2 = conv2d(a1, w2)
    a2 = relu(z2)
    
    a3 = maxpool(a2)
    
    z4 = w3 * a3
    a4 = relu(z4)
    
    z5 = w4 * a4
    a5 = softmax(z5)
    
    return a1, a2, a3, a4, a5
end

# Backward pass
function backward(x, y, a1, a2, a3, a4, a5, w1, w2, w3, w4, lr)
    dz5 = a5 - y
    dw4 = dz5 * a4'
    
    dz4 = (w4' * dz5) .* relu_prime(a4)
    dw3 = dz4 * a3'
    
    da3 = w3' * dz4
    dz2 = (unpool(da3, a2)) .* relu_prime(a2)
    dw2 = conv2d(a1, dz2, "valid")
    
    dz1 = (conv2d(dz2, rot180(w2), "full")) .* relu_prime(a1)
    dw1 = conv2d(x, dz1, "valid")
    
    # Update weights
    w1 -= lr * dw1
    w2 -= lr * dw2
    w3 -= lr * dw3
    w4 -= lr * dw4
    
    return w1, w2, w3, w4
end

# Example usage
x = rand(Float32, 28, 28, 1)  # Example input
y = onehot(3, 10)  # Example label

# Initialize weights
w1 = randn(Float32, 3, 3, 1, 16)
w2 = randn(Float32, 3, 3, 16, 32)
w3 = randn(Float32, 32*6*6, 128)
w4 = randn(Float32, 128, 10)

# Learning rate
lr = 0.001

# Forward pass
a1, a2, a3, a4, a5 = forward(x, w1, w2, w3, w4)

# Compute loss
loss = cross_entropy_loss(a5, y)

# Backward pass and weight updates
w1, w2, w3, w4 = backward(x, y, a1, a2, a3, a4, a5, w1, w2, w3, w4, lr)


# Code with more examples:
# Backward pass
function backward(x, y, a1, a2, a3, a4, a5, w1, w2, w3, w4, lr)
    dz5 = a5 - y
    dw4 = dz5 * a4'
    
    dz4 = (w4' * dz5) .* relu_prime(a4)
    dw3 = dz4 * a3'
    
    da3 = w3' * dz4
    dz2 = (unpool(da3, a2)) .* relu_prime(a2)
    dw2 = conv2d(a1, dz2, "valid")
    
    dz1 = (conv2d(dz2, rot180(w2), "full")) .* relu_prime(a1)
    dw1 = conv2d(x, dz1, "valid")
    
    # Update weights
    w1 -= lr * dw1
    w2 -= lr * dw2
    w3 -= lr * dw3
    w4 -= lr * dw4
    
    return w1, w2, w3, w4
end


# max pooling saving indices
function maxpool_forward(x, pool_size)
    (h, w, c, n) = size(x)
    ph, pw = pool_size
    out_h = div(h, ph)
    out_w = div(w, pw)
    
    pooled = zeros(Float32, out_h, out_w, c, n)
    indices = zeros(Int, out_h, out_w, c, n)
    
    for i in 1:out_h
        for j in 1:out_w
            for k in 1:c
                for l in 1:n
                    patch = x[(i-1)*ph+1:i*ph, (j-1)*pw+1:j*pw, k, l]
                    max_val, max_idx = findmax(patch)
                    pooled[i, j, k, l] = max_val
                    indices[i, j, k, l] = max_idx
                end
            end
        end
    end
    
    return pooled, indices
end

# max pooling backward (e.g.--unpooling)

function maxpool_backward(dout, indices, pool_size, input_size)
    (h, w, c, n) = input_size
    ph, pw = pool_size
    out_h = div(h, ph)
    out_w = div(w, pw)
    
    dinput = zeros(Float32, h, w, c, n)
    
    for i in 1:out_h
        for j in 1:out_w
            for k in 1:c
                for l in 1:n
                    max_idx = indices[i, j, k, l]
                    patch = dinput[(i-1)*ph+1:i*ph, (j-1)*pw+1:j*pw, k, l]
                    patch[max_idx] = dout[i, j, k, l]
                end
            end
        end
    end
    
    return dinput
end

# Example input
x = rand(Float32, 8, 8, 1, 1)

# Forward pass
pool_size = (2, 2)
pooled, indices = maxpool_forward(x, pool_size)

# Example gradient from the next layer
dout = rand(Float32, 4, 4, 1, 1)

# Backward pass
dinput = maxpool_backward(dout, indices, pool_size, size(x))

#=
Explanation:
Forward Pass:

maxpool_forward performs max pooling and saves the indices of the maximum values.
pooled is the output of the max pooling layer.
indices contains the indices of the maximum values in each pooling region.
Backward Pass:

maxpool_backward uses the saved indices to propagate the gradients back to the correct locations in the input.
dout is the gradient from the next layer.
dinput is the gradient with respect to the input of the max pooling layer.
=#

function rot180(kernel)
    return reverse(reverse(kernel, dims=1), dims=2)
end

function conv2d_rot180(input, kernel)
    (h, w) = size(input)
    (kh, kw) = size(kernel)
    output = zeros(Float32, h - kh + 1, w - kw + 1)
    
    for i in 1:(h - kh + 1)
        for j in 1:(w - kw + 1)
            sum = 0.0
            for ki in 1:kh
                for kj in 1:kw
                    sum += input[i + ki - 1, j + kj - 1] * kernel[kh - ki + 1, kw - kj + 1]
                end
            end
            output[i, j] = sum
        end
    end
    
    return output
end