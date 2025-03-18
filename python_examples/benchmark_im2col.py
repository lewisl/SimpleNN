import numpy as np
import time
from functools import partial

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

def traditional_conv(input_data, weights, bias):
    """Traditional nested-loop convolution implementation"""
    N, C_in, H, W = input_data.shape
    C_out, _, K, _ = weights.shape
    H_out = H - K + 1
    W_out = W - K + 1
    
    output = np.zeros((N, C_out, H_out, W_out))
    
    for n in range(N):
        for c_out in range(C_out):
            for h in range(H_out):
                for w in range(W_out):
                    for c_in in range(C_in):
                        output[n, c_out, h, w] += np.sum(
                            input_data[n, c_in, h:h+K, w:w+K] * weights[c_out, c_in]
                        )
    return output + bias[:, None, None]

def im2col_conv(input_data, weights, bias):
    """Matrix multiplication-based convolution using im2col"""
    N, C_in, H, W = input_data.shape
    C_out, _, K, _ = weights.shape
    H_out = H - K + 1
    W_out = W - K + 1
    
    # Transform input data into column format
    x_col = im2col(input_data, K, K, stride=1, pad=0)
    
    # Reshape weights and perform matrix multiplication
    w_col = weights.reshape(C_out, -1)
    out = np.dot(w_col, x_col) + bias[:, None]
    
    # Reshape output
    out = out.reshape(C_out, N, H_out, W_out)
    return out.transpose(1, 0, 2, 3)

def benchmark_convolutions():
    """Benchmark different convolution implementations"""
    # Test configurations
    configs = [
        # (batch_size, in_channels, height, width, out_channels, kernel_size)
        (32, 64, 56, 56, 128, 3),   # Typical middle ConvNet layer
        (8, 32, 112, 112, 64, 3),   # Early layer, smaller batch
        (64, 128, 28, 28, 256, 3),  # Later layer, larger batch
    ]
    
    results = []
    for config in configs:
        N, C_in, H, W, C_out, K = config
        
        # Generate test data
        input_data = np.random.randn(N, C_in, H, W)
        weights = np.random.randn(C_out, C_in, K, K)
        bias = np.random.randn(C_out)
        
        # Warm up cache
        traditional_conv(input_data[:1], weights[:1], bias[:1])
        im2col_conv(input_data[:1], weights[:1], bias[:1])
        
        # Time traditional convolution
        start = time.perf_counter()
        for _ in range(5):  # Multiple runs for more stable timing
            traditional_conv(input_data, weights, bias)
        trad_time = (time.perf_counter() - start) / 5
        
        # Time im2col convolution
        start = time.perf_counter()
        for _ in range(5):
            im2col_conv(input_data, weights, bias)
        im2col_time = (time.perf_counter() - start) / 5
        
        results.append({
            'config': config,
            'traditional_time': trad_time,
            'im2col_time': im2col_time,
            'speedup': trad_time / im2col_time
        })
        
        # Print results
        print(f"\nConfiguration: {config}")
        print(f"Traditional conv time: {trad_time:.3f}s")
        print(f"Im2col conv time: {im2col_time:.3f}s")
        print(f"Speedup: {trad_time/im2col_time:.2f}x")
    
    return results

# Run benchmarks
results = benchmark_convolutions()