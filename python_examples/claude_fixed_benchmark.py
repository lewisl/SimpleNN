import numpy as np
import time

def im2col(input_data, kernel_height, kernel_width, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - kernel_height) // stride + 1
    out_w = (W + 2 * pad - kernel_width) // stride + 1

    # Add padding if specified
    if pad > 0:
        pad_width = ((0,0), (0,0), (pad,pad), (pad,pad))
        input_data = np.pad(input_data, pad_width, mode='constant')
    
    # Initialize output matrix
    col = np.zeros((N * out_h * out_w, C * kernel_height * kernel_width))
    
    # Fill the column matrix
    for y in range(out_h):
        y_max = y * stride + kernel_height
        for x in range(out_w):
            x_max = x * stride + kernel_width
            # Extract the current window and reshape it into a column
            window = input_data[:, :, y*stride:y_max, x*stride:x_max]
            col[y*out_w + x::out_h*out_w] = window.reshape(N, -1)
            
    return col

def traditional_conv(input_data, weights, bias):
    """Traditional nested-loop convolution implementation"""
    N, C_in, H, W = input_data.shape
    C_out, _, K, _ = weights.shape
    H_out = H - K + 1
    W_out = W - K + 1
    
    output = np.zeros((N, C_out, H_out, W_out))
    
    for n in range(N):
        for c_out in range(C_out):
            for c_in in range(C_in):
                for h in range(H_out):
                    for w in range(W_out):
                        # Perform convolution for each input-output channel pair
                        output[n, c_out, h, w] += np.sum(
                            input_data[n, c_in, h:h+K, w:w+K] * weights[c_out, c_in]
                        )
    
    # Add bias to each output channel
    for c_out in range(C_out):
        output[:, c_out, :, :] += bias[c_out]
    
    return output

def im2col_conv(input_data, weights, bias):
    """Matrix multiplication-based convolution using im2col"""
    N, C_in, H, W = input_data.shape
    C_out, _, K, _ = weights.shape
    H_out = H - K + 1
    W_out = W - K + 1
    
    # Reshape input data into column format
    x_col = im2col(input_data, K, K)
    
    # Reshape weights into a matrix
    w_col = weights.reshape(C_out, -1)
    
    # Perform convolution as matrix multiplication
    out = np.dot(x_col, w_col.T)
    
    # Add bias
    out = out + bias
    
    # Reshape output back to proper format
    out = out.reshape(N, H_out, W_out, C_out)
    return out.transpose(0, 3, 1, 2)  # Change to NCHW format

def benchmark_convolutions():
    """Benchmark different convolution implementations"""
    print("Starting benchmarks...")
    
    # Test configurations - (batch_size, in_channels, height, width, out_channels, kernel_size)
    configs = [
        (4, 16, 32, 32, 32, 3),    # Small config for testing
        (8, 32, 28, 28, 64, 3),    # Medium config
        (16, 64, 14, 14, 128, 3),  # Larger config
    ]
    
    results = []
    for config in configs:
        N, C_in, H, W, C_out, K = config
        print(f"\nTesting configuration: batch={N}, in_ch={C_in}, H={H}, W={W}, out_ch={C_out}, kernel={K}")
        
        # Generate test data
        input_data = np.random.randn(N, C_in, H, W).astype(np.float32)
        weights = np.random.randn(C_out, C_in, K, K).astype(np.float32)
        bias = np.random.randn(C_out).astype(np.float32)
        
        # Warm up cache
        _ = traditional_conv(input_data[:1], weights[:1], bias[:1])
        _ = im2col_conv(input_data[:1], weights[:1], bias[:1])
        
        # Time traditional convolution
        start = time.perf_counter()
        for _ in range(3):  # Reduced number of runs for faster testing
            out1 = traditional_conv(input_data, weights, bias)
        trad_time = (time.perf_counter() - start) / 3
        
        # Time im2col convolution
        start = time.perf_counter()
        for _ in range(3):
            out2 = im2col_conv(input_data, weights, bias)
        im2col_time = (time.perf_counter() - start) / 3
        
        # Verify results match (within numerical precision)
        max_diff = np.max(np.abs(out1 - out2))
        
        results.append({
            'config': config,
            'traditional_time': trad_time,
            'im2col_time': im2col_time,
            'speedup': trad_time / im2col_time,
            'max_diff': max_diff
        })
        
        # Print results
        print(f"Traditional conv time: {trad_time:.3f}s")
        print(f"Im2col conv time: {im2col_time:.3f}s")
        print(f"Speedup: {trad_time/im2col_time:.2f}x")
        print(f"Max difference between outputs: {max_diff:.2e}")
    
    return results

if __name__ == "__main__":
    results = benchmark_convolutions()