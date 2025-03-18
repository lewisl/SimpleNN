function upsample_gradients(dL_dout::Array{Float32,4}, stride::Int)
    """
    Insert zeros between gradient values to handle stride > 1
    Input shape: (out_h, out_w, c_out, batch)
    Output shape: ((out_h-1)*stride + 1, (out_w-1)*stride + 1, c_out, batch)
    """
    h_out, w_out, c_out, batch = size(dL_dout)
    h_up = (h_out-1)*stride + 1
    w_up = (w_out-1)*stride + 1
    
    # Initialize upsampled gradient with zeros
    grad_up = zeros(Float32, h_up, w_up, c_out, batch)
    
    # Place original gradients with stride spacing
    grad_up[1:stride:end, 1:stride:end, :, :] = dL_dout
    
    return grad_up
end

function backward_conv_detailed(dL_dout::Array{Float32,4}, 
                              x_col::Matrix{Float32},
                              weights::Array{Float32,4},
                              stride::Int,
                              pad::Int)
    """
    Detailed backward pass showing dimension handling
    dL_dout: gradient from next layer (out_h, out_w, c_out, batch)
    x_col: cached input in column format
    weights: original weights (kernel_h, kernel_w, c_in, c_out)
    """
    kernel_h, kernel_w, c_in, c_out = size(weights)
    
    # 1. Handle striding by upsampling gradients
    if stride > 1
        dL_dout = upsample_gradients(dL_dout, stride)
    end
    
    # 2. Pad gradients for full convolution
    # Need padding to ensure output matches input size
    pad_h = kernel_h - 1
    pad_w = kernel_w - 1
    dL_dout_padded = pad_array(dL_dout, pad_h)
    
    # 3. Reshape gradient to column format
    # Shape: (c_out, batch * grad_h * grad_w)
    dL_dout_col = reshape(permutedims(dL_dout_padded, (3,4,1,2)), 
                         c_out, :)
    
    # 4. Compute weight gradients
    # x_col shape: (kernel_h * kernel_w * c_in, batch * out_h * out_w)
    # dL_dout_col shape: (c_out, batch * out_h * out_w)
    dL_dw = reshape(dL_dout_col * x_col',
                   kernel_h, kernel_w, c_in, c_out)
    
    # 5. Compute input gradients
    # First flip weights for correlation
    w_flipped = reverse(reverse(weights, dims=1), dims=2)
    w_col = reshape(permutedims(w_flipped, (1,2,4,3)), 
                   kernel_h * kernel_w * c_out, c_in)
    
    # Compute input gradients
    dL_dx_col = w_col * dL_dout_col
    
    # Reshape back to spatial format if needed
    # (Only needed for visualization or if converting back to im format)
    
    return dL_dx_col, dL_dw
end

# Example usage showing dimensions
function demonstrate_dimensions()
    # Example dimensions
    h, w = 28, 28          # Input dimensions
    c_in, c_out = 3, 16    # Channel dimensions
    batch = 32             # Batch size
    kernel_size = 3        # Kernel dimensions
    stride = 2             # Stride > 1 case
    pad = 1               # Padding
    
    # Calculate output dimensions
    out_h = (h + 2pad - kernel_size)÷stride + 1
    out_w = (w + 2pad - kernel_size)÷stride + 1
    
    println("Forward pass dimensions:")
    println("Input: ($h, $w, $c_in, $batch)")
    println("Weights: ($kernel_size, $kernel_size, $c_in, $c_out)")
    println("Output: ($out_h, $out_w, $c_out, $batch)")
    
    # Col format dimensions
    col_rows = kernel_size * kernel_size * c_in
    col_cols = batch * out_h * out_w
    println("\nCol format dimensions:")
    println("x_col: ($col_rows, $col_cols)")
    
    # Backward pass dimensions
    grad_up_h = (out_h-1)*stride + 1
    grad_up_w = (out_w-1)*stride + 1
    println("\nBackward pass dimensions:")
    println("Initial gradient: ($out_h, $out_w, $c_out, $batch)")
    println("After upsampling: ($grad_up_h, $grad_up_w, $c_out, $batch)")
    println("After padding: ($(grad_up_h+2*(kernel_size-1)), ",
            "$(grad_up_w+2*(kernel_size-1)), $c_out, $batch)")
end