function backward_conv(layer::ConvLayer, dL_dout::Array{Float64,4}, input_img_size::ImgSize, conv_inputs::ConvInputs)
    x_col = conv_inputs.x_col
    w_col = conv_inputs.w_col
    kernel_size = size(layer.weights)[1]
    
    # Use same im2col function for gradient rearrangement
    dL_dout_col = im2col(dL_dout, kernel_size, layer.stride, layer.pad)
    
    # Rest of computation...
    dL_dw_col = dL_dout_col * x_col'
    dL_dx_col = w_col * dL_dout_col
    dL_dx = col2im(dL_dx_col, input_img_size, kernel_size, layer.stride, layer.pad)
    
    return dL_dx, dL_dw, dL_db
end