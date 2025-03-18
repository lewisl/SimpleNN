
using Statistics
using LinearAlgebra

# Layer type definitions
struct ConvLayer
    weights::Array{Float64,4}  # (kernel_h, kernel_w, in_channels, out_channels)
    bias::Vector{Float64}
    stride::Int
    pad::Int
end

struct PoolLayer
    size::Int
    stride::Int
end

struct FCLayer
    weights::Matrix{Float64}  # (out_features, in_features)
    bias::Vector{Float64}
end

# Forward pass

# TODO handle asymmetric padding
function pad_array(x::Array{Float64,4}, pad::Int)
    batch_size, height, width, channels = size(x)
    result = zeros(Float64, batch_size, height + 2pad, width + 2pad, channels)
    result[:, pad+1:end-pad, pad+1:end-pad, :] = x
    return result
end

function im2col(x::Array{Float64,4}, kernel_size::Int, stride::Int, pad::Int)
    batch_size, height, width, in_channels = size(x)
    
    # Add padding if needed
    if pad > 0
        x = pad_array(x, pad)
    end
    
    # Calculate output dimensions
    out_height = div((height + 2pad - kernel_size), stride) + 1
    out_width = div((width + 2pad - kernel_size), stride) + 1
    
    # Initialize column matrix
    col = zeros(Float64, kernel_size * kernel_size * in_channels, 
                        batch_size * out_height * out_width)
    # @show size(col)
    
    # Fill column matrix
    col_idx = 1
    for b in 1:batch_size
        for i in 1:out_height
            for j in 1:out_width
                h_start = (i-1) * stride + 1
                h_end = h_start + kernel_size - 1
                w_start = (j-1) * stride + 1
                w_end = w_start + kernel_size - 1
                
                # Extract patch
                patch = view(x, b:b, h_start:h_end, w_start:w_end, :)
                # @show size(patch)
                col[:, col_idx] = vec(patch)
                col_idx += 1
            end
        end
    end
    
    return col
end



function col2im(col::Array{Float64,2}, x_shape::Tuple, kernel_size::Int, 
                stride::Int, pad::Int)
    height, width, channels, batch_size = x_shape
    out_height = div((height + 2pad - kernel_size), stride) + 1
    out_width = div((width + 2pad - kernel_size), stride) + 1
    
    # Initialize output array with padding
    x_padded = zeros(Float64, batch_size, height + 2pad, width + 2pad, channels)
    
    col_idx = 1
    for b in 1:batch_size
        for i in 1:out_height
            for j in 1:out_width
                h_start = (i-1) * stride + 1
                h_end = h_start + kernel_size - 1
                w_start = (j-1) * stride + 1
                w_end = w_start + kernel_size - 1
                
                # Reshape column back to patch
                patch = reshape(col[:, col_idx], 
                                kernel_size, kernel_size, channels)
                
                # Manual addition instead of broadcasting
                for c in 1:channels
                    for h in 1:kernel_size
                        for w in 1:kernel_size
                            x_padded[b, h_start+h-1, w_start+w-1, c] += 
                                patch[h, w, c]
                        end
                    end
                end
                
                col_idx += 1
            end
        end
    end
    
    # Remove padding if needed
    if pad > 0
        return x_padded[:, pad+1:end-pad, pad+1:end-pad, :]
    end
    return x_padded
end


function forward_conv(layer::ConvLayer, x::Array{Float64,4})
    x_col = im2col(x, size(layer.weights)[1], layer.stride, layer.pad)
    w_col = reshape(layer.weights, :, size(layer.weights)[4])
    
    out = w_col' * x_col
    # Reshape to (out_h, out_w, out_channels, batch)
    out_h = div((size(x,1) - size(layer.weights, 1)), layer.stride) + 1
    out_w = div((size(x,2) - size(layer.weights, 2)), layer.stride) + 1
    out = reshape(permutedims(reshape(out, size(layer.weights)[4], :, size(x)[4]), 
                            (2,1,3)), out_h, out_w, :, size(x)[4])

    # Add bias:  TODO faster way to do this without reshape and permutedims
    for i in 1:out_channels
        out[:,:,:,i] .+= layer.bias[i]
    end
    
    return out, (x_col, w_col)
end

function forward_pool(layer::PoolLayer, x::Array{Float64,4})
    h, w, c, batch = size(x)
    out_h = (h - layer.size) ÷ layer.stride + 1
    out_w = (w - layer.size) ÷ layer.stride + 1
    
    out = zeros(Float64, out_h, out_w, c, batch)
    max_indices = zeros(Int, out_h, out_w, c, batch)
    
    for b in 1:batch, c in 1:c
        for i in 1:out_h, j in 1:out_w
            h_start = (i-1) * layer.stride + 1
            w_start = (j-1) * layer.stride + 1
            window = view(x, h_start:h_start+layer.size-1,
                        w_start:w_start+layer.size-1, c, b)
            out[i,j,c,b], idx = findmax(window)
            max_indices[i,j,c,b] = idx
        end
    end
    
    return out, max_indices
end

function forward_fc(layer::FCLayer, x::Matrix{Float64})
    return layer.weights * x .+ layer.bias, x
end

# Backward passes
function backward_conv(layer::ConvLayer, dL_dout::Array{Float64,4}, cache)
    x, x_col, w_col = cache  # what is in cache? x, x_col, w_col? or no x?
    batch_size, height, width, in_channels = size(x)
    kernel_size = size(layer.weights)[1]
    out_channels = size(layer.weights)[4]

    # Reshape incoming gradient for matrix operations
    dL_dout_reshaped = reshape(permutedims(dL_dout, (4,1,2,3)), out_channels, :)
    
    # Calculate weight gradients
    dL_dw_col = dL_dout_reshaped * x_col'
    dL_db = vec(sum(dL_dout_reshaped, dims=(1,2,3)))
    
    # Calculate gradient for previous layer
    dL_dx_col = w_col * dL_dout_reshaped
    
    return dL_dx_col, dL_dw, dL_db # should these be col or im format?
end

function backward_pool(layer::PoolLayer, dL_dout::Array{Float64,4}, 
                        x_shape::Tuple, max_indices::Array{Int,4})
    h, w, c, batch = x_shape
    dL_dx = zeros(Float64, h, w, c, batch)
    
    out_h, out_w = size(dL_dout)[1:2]
    
    for b in 1:batch, c in 1:c
        for i in 1:out_h, j in 1:out_w
            h_start = (i-1) * layer.stride + 1
            w_start = (j-1) * layer.stride + 1
            idx = max_indices[i,j,c,b]
            h_offset, w_offset = ind2sub((layer.size, layer.size), idx)
            dL_dx[h_start+h_offset-1, w_start+w_offset-1, c, b] = dL_dout[i,j,c,b]
        end
    end
    
    return dL_dx
end

function backward_fc(layer::FCLayer, dL_dout::Matrix{Float64}, cache)
    x = cache
    dL_dw = dL_dout * x'
    dL_db = vec(sum(dL_dout, dims=2))
    dL_dx = layer.weights' * dL_dout
    return dL_dx, dL_dw, dL_db
end

# Transition functions
function flatten_conv_output(x::Array{Float64,4})
    # Reshape from (h, w, c, batch) to (h*w*c, batch)
    return reshape(permutedims(x, (1,2,3,4)), :, size(x)[4])
end

function unflatten_gradient(grad::Matrix{Float64}, conv_shape::Tuple)
    # Reshape from (h*w*c, batch) to (h, w, c, batch)
    h, w, c, batch = conv_shape
    return permutedims(reshape(grad, h, w, c, batch), (1,2,3,4))
end

# Full forward and backward for a simple CNN
function forward_cnn(layers, x)
    # Assuming layers is a tuple of (conv1, pool1, conv2, pool2, fc1, fc2)
    caches = []
    
    # First conv + pool
    out1, cache1 = forward_conv(layers[1], x)
    out1_pool, cache1_pool = forward_pool(layers[2], out1)
    push!(caches, (cache1, cache1_pool, size(out1)))
    
    # Second conv + pool
    out2, cache2 = forward_conv(layers[3], out1_pool)
    out2_pool, cache2_pool = forward_pool(layers[4], out2)
    push!(caches, (cache2, cache2_pool, size(out2)))
    
    # Flatten and FC layers
    flattened = flatten_conv_output(out2_pool)
    out_fc1, cache_fc1 = forward_fc(layers[5], flattened)
    out_fc2, cache_fc2 = forward_fc(layers[6], out_fc1)
    push!(caches, (cache_fc1, cache_fc2))
    
    return out_fc2, caches
end

function backward_cnn(layers, dL_dout, caches)
    gradients = []
    
    # Assuming dL_dout is (num_classes, batch) from softmax gradient
    # Backward through FC layers
    (cache_fc1, cache_fc2) = caches[3]
    dx_fc2, dw_fc2, db_fc2 = backward_fc(layers[6], dL_dout, cache_fc2)
    dx_fc1, dw_fc1, db_fc1 = backward_fc(layers[5], dx_fc2, cache_fc1)
    push!(gradients, (dw_fc2, db_fc2, dw_fc1, db_fc1))
    
    # Unflatten gradient for conv layers
    (cache2, cache2_pool, conv2_shape) = caches[2]
    dx_unflatten = unflatten_gradient(dx_fc1, conv2_shape)
    
    # Backward through second conv+pool
    dx_pool2 = backward_pool(layers[4], dx_unflatten, conv2_shape, cache2_pool)
    dx_conv2, dw_conv2, db_conv2 = backward_conv(layers[3], dx_pool2, cache2)
    
    # Backward through first conv+pool
    (cache1, cache1_pool, conv1_shape) = caches[1]
    dx_pool1 = backward_pool(layers[2], dx_conv2, conv1_shape, cache1_pool)
    dx_conv1, dw_conv1, db_conv1 = backward_conv(layers[1], dx_pool1, cache1)
    
    push!(gradients, (dw_conv2, db_conv2, dw_conv1, db_conv1))
    
    return gradients
end

#=
Key points about this implementation:

1. Layer Transitions:
   - Conv layers maintain 4D format (h, w, c, batch)
   - Pooling preserves channels but reduces spatial dimensions
   - Flattening converts to 2D (features, batch) for FC layers
   - Gradients must be reshaped when transitioning back

2. Pooling Layer:
   - Tracks max indices for efficient backprop
   - Reduces spatial dimensions while preserving channels
   - Backprop routes gradients only through max positions

3. Dimensional Handling:
   - Initial dL_dout from softmax is (num_classes, batch)
   - Gets expanded back to conv dimensions through FC layers
   - Maintains proper shapes through all transformations

Would you like me to:
1. Add the training loop with loss calculation?
2. Show how to handle batch normalization between layers?
3. Add activation functions (ReLU) explicitly?
=#