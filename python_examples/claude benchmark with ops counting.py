import numpy as np
import time

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

class OpCounter:
    """Tracks floating point operations (multiply-adds)"""
    def __init__(self):
        self.total_ops = 0
        self.start_time = None
        self.end_time = None
    
    def add_ops(self, count):
        self.total_ops += count
    
    def start_timing(self):
        self.start_time = time.perf_counter()
    
    def end_timing(self):
        self.end_time = time.perf_counter()
    
    @property
    def elapsed_time(self):
        return self.end_time - self.start_time
    
    @property
    def gflops(self):
        """Calculate GFLOPS (billion floating point operations per second)"""
        return (self.total_ops / self.elapsed_time) / 1e9

def traditional_conv_counted(input_data, weights, bias):
    """Traditional convolution implementation with operation counting"""
    counter = OpCounter()
    counter.start_timing()
    
    N, C_in, H, W = input_data.shape
    C_out, _, K, _ = weights.shape
    H_out = H - K + 1
    W_out = W - K + 1
    
    output = np.zeros((N, C_out, H_out, W_out))
    
    # Count operations:
    # For each output element: K*K multiply-adds per input channel
    ops_per_output = 2 * K * K * C_in  # multiply-add counts as 2 ops
    total_outputs = N * C_out * H_out * W_out
    counter.add_ops(ops_per_output * total_outputs)
    
    # Actual computation
    for n in range(N):
        for c_out in range(C_out):
            for c_in in range(C_in):
                for h in range(H_out):
                    for w in range(W_out):
                        output[n, c_out, h, w] += np.sum(
                            input_data[n, c_in, h:h+K, w:w+K] * weights[c_out, c_in]
                        )
    
    # Add bias - one add per output element
    counter.add_ops(total_outputs)
    for c_out in range(C_out):
        output[:, c_out, :, :] += bias[c_out]
    
    counter.end_timing()
    return output, counter

def im2col_conv_counted(input_data, weights, bias):
    """Matrix multiplication-based convolution with operation counting"""
    counter = OpCounter()
    counter.start_timing()
    
    N, C_in, H, W = input_data.shape
    C_out, _, K, _ = weights.shape
    H_out = H - K + 1
    W_out = W - K + 1
    
    # Transform input data into column format
    x_col = im2col(input_data, K, K)
    w_col = weights.reshape(C_out, -1)
    
    # Count matrix multiplication operations:
    # For each element in result: C_in*K*K multiply-adds
    m, k = x_col.shape
    k, n = k, w_col.shape[0]
    counter.add_ops(2 * m * n * k)  # multiply-add counts as 2 ops
    
    # Perform convolution as matrix multiplication
    out = np.dot(x_col, w_col.T)
    
    # Add bias - one add per output element
    counter.add_ops(out.size)
    out = out + bias
    
    # Reshape output
    out = out.reshape(N, H_out, W_out, C_out)
    out = out.transpose(0, 3, 1, 2)
    
    counter.end_timing()
    return out, counter

def run_comparison():
    """Run both implementations and compare operations and performance"""
    # Test configuration
    N, C_in, H, W = 16, 64, 28, 28  # Input shape
    C_out, K = 128, 3               # Output channels, kernel size
    
    # Generate test data
    input_data = np.random.randn(N, C_in, H, W).astype(np.float32)
    weights = np.random.randn(C_out, C_in, K, K).astype(np.float32)
    bias = np.random.randn(C_out).astype(np.float32)
    
    # Run both implementations
    trad_out, trad_counter = traditional_conv_counted(input_data, weights, bias)
    im2col_out, im2col_counter = im2col_conv_counted(input_data, weights, bias)
    
    # Print detailed results
    print(f"\nOperation Count Comparison:")
    print(f"{'':20} {'Traditional':>15} {'Im2col':>15}")
    print("-" * 50)
    print(f"Total Operations:{trad_counter.total_ops:15,} {im2col_counter.total_ops:15,}")
    print(f"Time (seconds):{trad_counter.elapsed_time:15.4f} {im2col_counter.elapsed_time:15.4f}")
    print(f"GFLOPS:{trad_counter.gflops:15.2f} {im2col_counter.gflops:15.2f}")
    
    # Verify results match
    max_diff = np.max(np.abs(trad_out - im2col_out))
    print(f"\nMaximum difference between outputs: {max_diff:.2e}")
    
    # Calculate speedup
    speedup = trad_counter.elapsed_time / im2col_counter.elapsed_time
    print(f"\nSpeedup: {speedup:.1f}x")
    
    # Calculate efficiency (operations per second)
    trad_efficiency = trad_counter.gflops
    im2col_efficiency = im2col_counter.gflops
    print(f"\nEfficiency comparison:")
    print(f"Traditional: {trad_efficiency:.2f} GFLOPS")
    print(f"Im2col: {im2col_efficiency:.2f} GFLOPS")
    print(f"Efficiency ratio: {im2col_efficiency/trad_efficiency:.1f}x")

if __name__ == "__main__":
    run_comparison()