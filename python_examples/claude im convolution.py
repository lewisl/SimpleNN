import numpy as np

def im2col(input_data, kernel_height, kernel_width, stride=1, pad=1):
    """
    Transform image-style data into columnar format for efficient convolution.
    
    Args:
        input_data: Array of shape (N, C, H, W)
        kernel_height, kernel_width: Dimensions of the convolution kernel
        stride: Convolution stride
        pad: Padding size
    
    Returns:
        col: Columnar data ready for convolution
        col_shape: Original shape information for col2im
    """
    N, C, H, W = input_data.shape
    
    # Calculate output dimensions
    out_h = (H + 2 * pad - kernel_height) // stride + 1
    out_w = (W + 2 * pad - kernel_width) // stride + 1
    
    # Add padding to input
    img = np.pad(input_data, [(0,0), (0,0), (pad,pad), (pad,pad)], 'constant')
    
    # Prepare indices for sliding window view
    col = np.zeros((N, C, kernel_height, kernel_width, out_h, out_w))
    for y in range(kernel_height):
        y_max = y + stride * out_h
        for x in range(kernel_width):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    
    # Reshape to columnar format
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col, (N, C, H, W, out_h, out_w)

def col2im(col, col_shape, kernel_height, kernel_width, stride=1, pad=1):
    """
    Transform columnar data back to image format.
    
    Args:
        col: Columnar data
        col_shape: Original shape information from im2col
        kernel_height, kernel_width: Dimensions of the convolution kernel
        stride: Convolution stride
        pad: Padding size
    
    Returns:
        img: Array of shape (N, C, H, W)
    """
    N, C, H, W, out_h, out_w = col_shape
    
    # Reshape column data back to intermediate format
    col = col.reshape(N, out_h, out_w, C, kernel_height, kernel_width)
    col = col.transpose(0, 3, 4, 5, 1, 2)
    
    # Initialize output array
    img = np.zeros((N, C, H + 2 * pad, W + 2 * pad))
    
    # Accumulate values
    for y in range(kernel_height):
        y_max = y + stride * out_h
        for x in range(kernel_width):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
    
    # Remove padding
    return img[:, :, pad:H + pad, pad:W + pad]

def conv_backward_im(dL_dout, x, w, stride=1, pad=0):
    """
    Backward pass for convolutional layer keeping data in im (image) format.
    
    Args:
        dL_dout: Gradient of loss with respect to conv output, shape (N, C_out, H_out, W_out)
        x: Input data, shape (N, C_in, H, W)
        w: Convolution weights, shape (C_out, C_in, K, K)
        stride: Convolution stride
        pad: Padding size
    
    Returns:
        dL_dx: Gradient with respect to input x
        dL_dw: Gradient with respect to weights w
        dL_db: Gradient with respect to bias
    """
    N, C_in, H, W = x.shape
    C_out, _, K, _ = w.shape
    _, _, H_out, W_out = dL_dout.shape
    
    # First pad the input if needed
    if pad > 0:
        x_pad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), 'constant')
    else:
        x_pad = x
    
    # Initialize gradients
    dL_dx_pad = np.zeros_like(x_pad)
    dL_dw = np.zeros_like(w)
    
    # For each output position, accumulate gradients from all affected input positions
    for n in range(N):
        for c_out in range(C_out):
            for c_in in range(C_in):
                for h_out in range(H_out):
                    h_start = h_out * stride
                    for w_out in range(W_out):
                        w_start = w_out * stride
                        
                        # Current gradient at this output position
                        grad = dL_dout[n, c_out, h_out, w_out]
                        
                        # Update weight gradients
                        # For each weight, multiply input patch by output gradient
                        dL_dw[c_out, c_in, :, :] += x_pad[n, c_in, 
                                                         h_start:h_start+K, 
                                                         w_start:w_start+K] * grad
                        
                        # Update input gradients
                        # For each input in the patch, multiply by corresponding weight
                        dL_dx_pad[n, c_in,
                                 h_start:h_start+K,
                                 w_start:w_start+K] += w[c_out, c_in, :, :] * grad
    
    # Remove padding from dx if needed
    if pad > 0:
        dL_dx = dL_dx_pad[:, :, pad:-pad, pad:-pad]
    else:
        dL_dx = dL_dx_pad
    
    # Bias gradient is sum over all except channel dimension
    dL_db = dL_dout.sum(axis=(0,2,3))
    
    return dL_dx, dL_dw, dL_db

def conv_backward_col(dL_dout, x_col, w_col, x_shape, w_shape, stride=1, pad=0):
    """
    Backward pass for convolutional layer using column format.
    Much more efficient than im format due to matrix operations.
    
    Args:
        dL_dout: Gradient of loss with respect to conv output (N, C_out, H_out, W_out)
        x_col: Input data in column format from forward pass
        w_col: Reshaped weights from forward pass
        x_shape: Original input shape (N, C_in, H, W)
        w_shape: Original weight shape (C_out, C_in, K, K)
        stride: Convolution stride
        pad: Padding size
    """
    N, C_in, H, W = x_shape
    C_out, _, K, _ = w_shape
    
    # Reshape dL_dout to match column format
    dL_dout_reshaped = dL_dout.transpose(1, 2, 3, 0).reshape(C_out, -1)
    
    # Compute weight gradients using matrix multiplication
    dL_dw = np.dot(dL_dout_reshaped, x_col.T)
    dL_dw = dL_dw.reshape(w_shape)
    
    # Compute input gradients using matrix multiplication
    dL_dx_col = np.dot(w_col.T, dL_dout_reshaped)
    
    # Convert input gradients back to image format
    dL_dx = col2im(dL_dx_col, x_shape, K, K, stride, pad)
    
    # Bias gradient is sum over all except channel dimension
    dL_db = dL_dout.sum(axis=(0,2,3))
    
    return dL_dx, dL_dw, dL_db

# Example usage and verification
def verify_backward_passes():
    """Compare results of both backward pass implementations"""
    # Setup test data
    N, C_in, H, W = 2, 3, 4, 4  # Small size for verification
    C_out, K = 2, 3
    stride, pad = 1, 1
    
    # Create random input and weights
    x = np.random.randn(N, C_in, H, W)
    w = np.random.randn(C_out, C_in, K, K)
    dL_dout = np.random.randn(N, C_out, H_out, W_out)
    
    # Run both implementations
    dx_im, dw_im, db_im = conv_backward_im(dL_dout, x, w, stride, pad)
    
    # For col version, need forward pass data
    x_col = im2col(x, K, K, stride, pad)
    w_col = w.reshape(C_out, -1)
    dx_col, dw_col, db_col = conv_backward_col(dL_dout, x_col, w_col, 
                                             x.shape, w.shape, stride, pad)
    
    # Compare results
    print("Maximum differences between implementations:")
    print(f"dx diff: {np.abs(dx_im - dx_col).max():.2e}")
    print(f"dw diff: {np.abs(dw_im - dw_col).max():.2e}")
    print(f"db diff: {np.abs(db_im - db_col).max():.2e}")