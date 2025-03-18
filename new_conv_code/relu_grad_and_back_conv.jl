# ReLU gradient - simple element-wise operation
function relu_backward(dL_dout::Array{Float64,4}, conv_output::Array{Float64,4})
    return dL_dout .* Float64.(conv_output .> 0)
end

# Convolution backward using im2col
function conv_backward(dL_dout::Array{Float64,4}, layer::ConvLayer, 
                        input_img_size::ImgSize, conv_inputs::ConvInputs)
    # Unpack saved inputs from forward pass
    x_col = conv_inputs.x_col  # Already includes padding handling
    w_col = conv_inputs.w_col
    
    # Unpack dimensions
    kernel_size = size(layer.weights)[1]
    out_channels = size(layer.weights)[4]
    
    # Reshape gradient for matrix multiplication
    dL_dout_reshaped = reshape(dL_dout, out_channels, :)
    
    # Compute weight gradient using matrix multiplication
    dL_dw_col = dL_dout_reshaped * x_col'
    dL_dw = reshape(dL_dw_col, kernel_size, kernel_size, input_img_size.channels, out_channels)
    
    # Compute input gradient
    dL_dx_col = w_col * dL_dout_reshaped
    dL_dx = col2im(dL_dx_col, input_img_size, kernel_size, layer.stride, layer.pad)
    
    # Compute bias gradient - sum across spatial dims and batch
    dL_db = vec(sum(dL_dout; dims=(1,2,4)))
    
    return dL_dx, dL_dw, dL_db
end

# Combined backward pass
function conv_layer_backward(dL_dout::Array{Float64,4}, conv_output::Array{Float64,4},
                           layer::ConvLayer, input_img_size::ImgSize, conv_inputs::ConvInputs)
    # First compute ReLU gradient
    dL_drelu = relu_backward(dL_dout, conv_output)
    
    # Then compute convolution gradients
    dL_dx, dL_dw, dL_db = conv_backward(dL_drelu, layer, input_img_size, conv_inputs)
    
    return dL_dx, dL_dw, dL_db
end