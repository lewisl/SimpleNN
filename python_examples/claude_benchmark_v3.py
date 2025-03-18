import numpy as np
import time
from collections import defaultdict

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

def detailed_benchmark(input_shape, kernel_shape, runs=5):
    """
    Detailed benchmark with extensive verification and timing
    
    Args:
        input_shape: Tuple of (N, C, H, W)
        kernel_shape: Tuple of (C_out, C_in, K, K)
        runs: Number of timing runs
    """
    results = defaultdict(list)
    
    # Generate test data
    input_data = np.random.randn(*input_shape).astype(np.float32)
    weights = np.random.randn(*kernel_shape).astype(np.float32)
    bias = np.random.randn(kernel_shape[0]).astype(np.float32)
    
    def run_with_timing(func, name):
        # Warmup
        _ = func(input_data, weights, bias)
        
        # Timing runs
        times = []
        outputs = []
        for _ in range(runs):
            start = time.perf_counter()
            out = func(input_data, weights, bias)
            end = time.perf_counter()
            times.append(end - start)
            outputs.append(out)
            
            # Memory stats (approximate)
            mem_used = out.nbytes + input_data.nbytes + weights.nbytes
            results[f'{name}_memory'].append(mem_used / 1024 / 1024)  # MB
        
        results[f'{name}_times'] = times
        return outputs[-1]  # Return last output for verification
    
    # Run both implementations
    print(f"\nRunning benchmark with shape: {input_shape} -> {kernel_shape}")
    trad_out = run_with_timing(traditional_conv, 'traditional')
    im2col_out = run_with_timing(im2col_conv, 'im2col')
    
    # Verify results
    max_diff = np.max(np.abs(trad_out - im2col_out))
    mean_diff = np.mean(np.abs(trad_out - im2col_out))
    
    # Print detailed results
    print("\nDetailed Results:")
    print(f"Traditional conv: {np.mean(results['traditional_times']):.6f}s ± {np.std(results['traditional_times']):.6f}s")
    print(f"Im2col conv: {np.mean(results['im2col_times']):.6f}s ± {np.std(results['im2col_times']):.6f}s")
    print(f"Speedup: {np.mean(results['traditional_times'])/np.mean(results['im2col_times']):.1f}x")
    print(f"\nOutput Verification:")
    print(f"Max difference: {max_diff:.2e}")
    print(f"Mean difference: {mean_diff:.2e}")
    print(f"Output shapes match: {trad_out.shape == im2col_out.shape}")
    print(f"Memory usage (MB):")
    print(f"  Traditional: {np.mean(results['traditional_memory']):.1f}")
    print(f"  Im2col: {np.mean(results['im2col_memory']):.1f}")
    
    return results

# Test with a variety of sizes
test_configs = [
    ((2, 16, 32, 32), (32, 16, 3, 3)),    # Small
    ((4, 32, 28, 28), (64, 32, 3, 3)),    # Medium
    ((8, 64, 14, 14), (128, 64, 3, 3)),   # Large
]

for input_shape, kernel_shape in test_configs:
    detailed_benchmark(input_shape, kernel_shape)