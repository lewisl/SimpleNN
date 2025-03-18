using LinearAlgebra

# Define our convolution layer structure
struct ConvLayerIm
    weights::Array{Float32,4}  # (kernel_height, kernel_width, in_channels, out_channels)
    bias::Vector{Float32}
    stride::Int
    pad::Int
end

# Forward pass using traditional image format
function forward_im(layer::ConvLayerIm, x::Array{Float32,4})
    # Extract dimensions from input and weights
    batch_size, height, width, in_channels = size(x)
    kernel_size = size(layer.weights)[1]  # Assuming square kernel
    out_channels = size(layer.weights)[4]
    
    # Calculate output dimensions with padding
    out_height = (height + 2layer.pad - kernel_size) ÷ layer.stride + 1
    out_width = (width + 2layer.pad - kernel_size) ÷ layer.stride + 1
    
    # Create padded input if needed
    if layer.pad > 0
        x_padded = zeros(Float32, batch_size, height + 2layer.pad, 
                        width + 2layer.pad, in_channels)
        x_padded[:, layer.pad+1:end-layer.pad, 
                    layer.pad+1:end-layer.pad, :] = x
    else
        x_padded = x
    end
    
    # Initialize output tensor
    out = zeros(Float32, batch_size, out_height, out_width, out_channels)
    
    # Perform convolution directly in image space
    for n in 1:batch_size
        for c_out in 1:out_channels
            for h in 1:out_height
                h_start = (h - 1) * layer.stride + 1
                h_end = h_start + kernel_size - 1
                
                for w in 1:out_width
                    w_start = (w - 1) * layer.stride + 1
                    w_end = w_start + kernel_size - 1
                    
                    # For each input channel
                    for c_in in 1:in_channels
                        # Extract the input patch and corresponding weights
                        patch = view(x_padded, n, h_start:h_end, w_start:w_end, c_in)
                        kernel = view(layer.weights, :, :, c_in, c_out)
                        
                        # Accumulate weighted sum
                        out[n, h, w, c_out] += sum(patch .* kernel)
                    end
                    
                    # Add bias
                    out[n, h, w, c_out] += layer.bias[c_out]
                end
            end
        end
    end
    
    return out, x_padded  # Cache padded input for backward pass
end

# Backward pass using traditional image format
function backward_im(layer::ConvLayerIm, dL_dout::Array{Float32,4}, 
                    x_padded::Array{Float32,4})
    # Get dimensions
    batch_size, out_height, out_width, out_channels = size(dL_dout)
    _, height_padded, width_padded, in_channels = size(x_padded)
    kernel_size = size(layer.weights)[1]
    
    # Original input dimensions
    height = height_padded - 2layer.pad
    width = width_padded - 2layer.pad
    
    # Initialize gradients
    dL_dx_padded = zeros(Float32, size(x_padded))
    dL_dw = zeros(Float32, size(layer.weights))
    dL_db = zeros(Float32, out_channels)
    
    # Compute gradients directly in image space
    for n in 1:batch_size
        for c_out in 1:out_channels
            for h in 1:out_height
                h_start = (h - 1) * layer.stride + 1
                h_end = h_start + kernel_size - 1
                
                for w in 1:out_width
                    w_start = (w - 1) * layer.stride + 1
                    w_end = w_start + kernel_size - 1
                    
                    # Current gradient at this output position
                    grad_value = dL_dout[n, h, w, c_out]
                    
                    # Update gradients for all input channels
                    for c_in in 1:in_channels
                        # Update weight gradients
                        patch = view(x_padded, n, h_start:h_end, w_start:w_end, c_in)
                        dL_dw[:, :, c_in, c_out] .+= patch .* grad_value
                        
                        # Update input gradients
                        kernel = view(layer.weights, :, :, c_in, c_out)
                        dL_dx_padded[n, h_start:h_end, w_start:w_end, c_in] .+= 
                            kernel .* grad_value
                    end
                    
                    # Accumulate bias gradients
                    dL_db[c_out] += grad_value
                end
            end
        end
    end
    
    # Remove padding from input gradients if needed
    if layer.pad > 0
        dL_dx = dL_dx_padded[:, layer.pad+1:end-layer.pad, 
                            layer.pad+1:end-layer.pad, :]
    else
        dL_dx = dL_dx_padded
    end
    
    return dL_dx, dL_dw, dL_db
end

# Test function
function test_conv_im()
    # Create small test case
    batch_size, height, width = 2, 6, 6
    in_channels, out_channels = 2, 3
    kernel_size = 3
    stride = 1
    pad = 1
    
    # Initialize layer
    weights = randn(Float32, kernel_size, kernel_size, in_channels, out_channels)
    bias = randn(Float32, out_channels)
    layer = ConvLayerIm(weights, bias, stride, pad)
    
    println("Testing convolution with:")
    println("Input shape: ($batch_size, $height, $width, $in_channels)")
    println("Kernel shape: ($kernel_size, $kernel_size, $in_channels, $out_channels)")
    
    # Forward pass
    x = randn(Float32, batch_size, height, width, in_channels)
    out, x_padded = forward_im(layer, x)
    println("\nForward pass output shape: ", size(out))
    
    # Backward pass
    dL_dout = randn(Float32, size(out))
    dL_dx, dL_dw, dL_db = backward_im(layer, dL_dout, x_padded)
    
    println("\nGradient shapes:")
    println("dL_dx: ", size(dL_dx))
    println("dL_dw: ", size(dL_dw))
    println("dL_db: ", size(dL_db))
    
    return true
end