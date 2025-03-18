using LinearAlgebra

# Type for holding convolution layer parameters
struct ConvLayer
    weights::Array{Float64,4}  # (height, width, in_channels, out_channels)
    bias::Vector{Float64}
    stride::Int
    pad::Int
end

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
    batch_size, height, width, channels = x_shape
    out_height = (height + 2pad - kernel_size) ÷ stride + 1
    out_width = (width + 2pad - kernel_size) ÷ stride + 1
    
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

function forward(layer::ConvLayer, x::Array{Float64,4})
    batch_size, height, width, in_channels = size(x)
    kernel_size = size(layer.weights)[1]
    out_channels = size(layer.weights)[4]
    
    out_height = (height + 2layer.pad - kernel_size) ÷ layer.stride + 1
    out_width = (width + 2layer.pad - kernel_size) ÷ layer.stride + 1
    
    # Convert input to column format
    x_col = im2col(x, kernel_size, layer.stride, layer.pad)
    # @show size(x_col)
    
    # Reshape weights for matrix multiplication
    w_col = reshape(layer.weights, kernel_size * kernel_size * in_channels, out_channels)
    # @show size(w_col)    

    # Perform convolution as matrix multiplication
    out = w_col' * x_col
    # @show size(out)
    
    # Reshape output
    out = reshape(out, out_channels, batch_size, out_height, out_width)
    out = permutedims(out, (2,3,4,1))  # Move channels to last dimension
    # @show size(out)
    
    # Add bias
    for i in 1:out_channels
        out[:,:,:,i] .+= layer.bias[i]
    end
    
    return out, (x, x_col, w_col)
end

function backward(layer::ConvLayer, dL_dout::Array{Float64,4}, cache::Tuple)
    x, x_col, w_col = cache
    batch_size, height, width, in_channels = size(x)
    kernel_size = size(layer.weights)[1]
    out_channels = size(layer.weights)[4]
    
    # Reshape gradient for matrix multiplication
    dL_dout_reshaped = reshape(permutedims(dL_dout, (4,1,2,3)), out_channels, :)
    
    # Compute gradients
    dL_dw_col = dL_dout_reshaped * x_col'
    dL_dw = reshape(dL_dw_col, kernel_size, kernel_size, in_channels, out_channels)
    
    dL_dx_col = w_col * dL_dout_reshaped
    # dL_dx = col2im(dL_dx_col, size(x), kernel_size, layer.stride, layer.pad)
    
    # Compute bias gradient
    dL_db = vec(sum(dL_dout; dims=(1,2,3)))
    
    return dL_dx, dL_dw, dL_db
end

# Test function
function test_conv()
    # Create small test case
    batch_size, height, width = 2, 6, 6  # Smaller size for testing
    in_channels, out_channels = 2, 3
    kernel_size = 3
    stride = 1
    pad = 1
    
    # Initialize layer
    weights = randn(Float64, kernel_size, kernel_size, in_channels, out_channels)
    bias = randn(Float64, out_channels)
    layer = ConvLayer(weights, bias, stride, pad)
    
    # Create input
    x = randn(Float64, batch_size, height, width, in_channels)
    
    # Forward pass
    # println("Running forward pass...")
    out, cache = forward(layer, x)
    # println("Forward pass output shape: ", size(out))
    
    # Backward pass
    # println("\nRunning backward pass...")
    dL_dout = randn(Float64, size(out))
    dL_dx, dL_dw, dL_db = backward(layer, dL_dout, cache)
    
    # println("\nGradient shapes:")
    # println("dL_dx: ", size(dL_dx))
    # println("dL_dw: ", size(dL_dw))
    # println("dL_db: ", size(dL_db))
    
    # Verify shapes
    # @assert size(dL_dx) == size(x) "Input gradient shape mismatch"
    # @assert size(dL_dw) == size(weights) "Weight gradient shape mismatch"
    # @assert size(dL_db) == size(bias) "Bias gradient shape mismatch"
    
    # println("\nAll shape checks passed!")
    return true
end

#=
The output from w_col' * x_col will have dimensions (out_channels × (batch_size * out_height * out_width)). So for each output channel i, we need to add the corresponding bias[i] to all columns in row i of this matrix.
This makes adding the bias pretty simple - we can just add the bias vector to each batch/spatial position in the output by broadcasting. In Julia this would be something like adding the bias (reshaped to match the output dimensions) to each column.
But let me double check my matrix dimensions:

x_col is (kernel_size * kernel_size * in_channels) × (batch_size * out_height * out_width)
w_col is (kernel_size * kernel_size * in_channels) × out_channels
w_col' * x_col gives us (out_channels × (batch_size * out_height * out_width))

So yes, each row i contains all the outputs for output channel i, spread across all batches and spatial positions. The bias for channel i needs to be added to all entries in row i.

=#