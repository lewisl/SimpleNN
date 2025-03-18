import numpy as np
import os
import subprocess
import platform
import time

def investigate_numpy_backend():
    """Investigate which backend NumPy is using for linear algebra operations"""
    
    print("NumPy Configuration:")
    print(f"NumPy version: {np.__version__}")
    print(f"Python version: {platform.python_version()}")
    
    # Check if we're running on Apple Silicon
    is_arm = platform.machine() == 'arm64'
    print(f"\nRunning on Apple Silicon: {is_arm}")
    
    # Show NumPy compilation information
    np_info = np.show_config()
    
    # Check for Accelerate framework
    def check_library_linking(lib_name):
        try:
            result = subprocess.run(
                ['otool', '-L', np.__file__],
                capture_output=True,
                text=True
            )
            return any(lib_name in line for line in result.stdout.split('\n'))
        except Exception:
            return "Could not determine"
    
    print("\nLibrary Dependencies:")
    print(f"Linked to Accelerate: {check_library_linking('Accelerate')}")
    
    # Benchmark to demonstrate BLAS acceleration
    def run_matmul_benchmark():
        sizes = [(100, 100), (1000, 1000), (2000, 2000)]
        print("\nMatrix Multiplication Benchmark:")
        print("Size\t\tTime (seconds)")
        for n in sizes:
            a = np.random.rand(n[0], n[1])
            b = np.random.rand(n[1], n[0])
            
            # Time the operation
            start = time.perf_counter()
            c = np.dot(a, b)
            duration = time.perf_counter() - start
            
            # Calculate theoretical FLOPS
            flops = 2 * n[0] * n[1] * n[0]  # multiply-add for each element
            gflops = flops / duration / 1e9
            
            print(f"{n[0]}x{n[1]}\t{duration:.4f}\t({gflops:.2f} GFLOPS)")
    
    run_matmul_benchmark()
    
    # Environment information
    print("\nEnvironment Variables affecting NumPy:")
    relevant_vars = [
        'NUMPY_THREADING',
        'MKL_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS',
        'OMP_NUM_THREADS'
    ]
    
    for var in relevant_vars:
        value = os.environ.get(var, 'Not set')
        print(f"{var}: {value}")

investigate_numpy_backend()