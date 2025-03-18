using Statistics
using Printf

# Structure to hold benchmark results
struct BenchmarkResult
    implementation::String
    batch_size::Int
    in_channels::Int
    out_channels::Int
    spatial_size::Int
    mean_time::Float64
    std_time::Float64
    mean_memory::Float64
    std_memory::Float64
    allocations::Int
    gflops::Float64
end

function count_conv_flops(N, C_in, C_out, H, W, K)
    # Calculate FLOPs for one convolution operation
    # Each output point requires K*K*C_in multiply-adds
    H_out = H - K + 1
    W_out = W - K + 1
    ops_per_point = 2 * K * K * C_in  # multiply-add counts as 2 ops
    total_points = N * C_out * H_out * W_out
    return ops_per_point * total_points
end

function run_single_benchmark(implementation, layer, input, warmup=3, iterations=10)
    times = Float64[]
    memories = Float64[]
    allocation_counts = Int[]
    
    # Warm up
    for _ in 1:warmup
        GC.gc()  # Clear garbage collection
        if implementation == "im"
            forward_im(layer, input)
        else
            forward(layer, input)
        end
    end
    
    # Actual benchmarking
    for _ in 1:iterations
        GC.gc()
        
        # Measure time and memory
        stats = @timed begin
            if implementation == "im"
                forward_im(layer, input)
            else
                forward(layer, input)
            end
        end
        
        push!(times, stats.time)
        push!(memories, stats.bytes / 1024)  # Convert to KB
        push!(allocation_counts, stats.gcstats.allocd)
    end
    
    # Calculate GFLOPS
    N, H, W, C_in = size(input)
    C_out = length(layer.bias)
    K = size(layer.weights)[1]
    flops = count_conv_flops(N, C_in, C_out, H, W, K)
    gflops = flops / (mean(times) * 1e9)
    
    return BenchmarkResult(
        implementation,
        N,
        C_in,
        C_out,
        H,
        mean(times),
        std(times),
        mean(memories),
        std(memories),
        mean(allocation_counts),
        gflops
    )
end

function benchmark_convolutions(;
    batch_sizes=[16, 32, 64],
    in_channels=[32, 64],
    out_channels=[32, 64],
    spatial_sizes=[28, 56],
    kernel_size=3,
    stride=1,
    pad=1,
    warmup=3,
    iterations=10
)
    results = BenchmarkResult[]
    
    for N in batch_sizes
        for C_in in in_channels
            for C_out in out_channels
                for H in spatial_sizes
                    println("\nBenchmarking configuration:")
                    println("Batch size: $N, Input channels: $C_in")
                    println("Output channels: $C_out, Spatial size: $H×$H")
                    
                    # Create test data
                    input = randn(Float32, N, H, H, C_in)
                    weights = randn(Float32, kernel_size, kernel_size, C_in, C_out)
                    bias = randn(Float32, C_out)
                    
                    # Create layers
                    layer_im = ConvLayerIm(weights, bias, stride, pad)
                    layer_col = ConvLayer(weights, bias, stride, pad)
                    
                    # Benchmark both implementations
                    result_im = run_single_benchmark("im", layer_im, input, 
                                                  warmup, iterations)
                    result_col = run_single_benchmark("col", layer_col, input, 
                                                   warmup, iterations)
                    
                    push!(results, result_im)
                    push!(results, result_col)
                    
                    # Print immediate results
                    print_comparison(result_im, result_col)
                end
            end
        end
    end
    
    return results
end

function print_comparison(im_result::BenchmarkResult, col_result::BenchmarkResult)
    println("\nDetailed Comparison:")
    println("─"^60)
    @printf("%-20s %-20s %-20s\n", "", "Image Format", "Column Format")
    println("─"^60)
    @printf("Mean Time (ms)     %-20.3f %-20.3f\n", 
            im_result.mean_time*1000, col_result.mean_time*1000)
    @printf("Std Time (ms)      %-20.3f %-20.3f\n", 
            im_result.std_time*1000, col_result.std_time*1000)
    @printf("Mean Memory (KB)   %-20.1f %-20.1f\n", 
            im_result.mean_memory, col_result.mean_memory)
    @printf("Allocations        %-20d %-20d\n", 
            im_result.allocations, col_result.allocations)
    @printf("GFLOPS             %-20.2f %-20.2f\n", 
            im_result.gflops, col_result.gflops)
    
    # Calculate speedup
    speedup = im_result.mean_time / col_result.mean_time
    memory_ratio = col_result.mean_memory / im_result.mean_memory
    @printf("\nSpeedup (col/im): %.2fx\n", speedup)
    @printf("Memory overhead (col/im): %.2fx\n", memory_ratio)
    println("─"^60)
end

function summarize_results(results::Vector{BenchmarkResult})
    println("\nOverall Performance Summary:")
    println("═"^80)
    
    # Group results by configuration
    configs = unique([(r.batch_size, r.in_channels, r.out_channels, r.spatial_size) 
                     for r in results])
    
    for config in configs
        N, C_in, C_out, H = config
        println("\nConfiguration: N=$N, C_in=$C_in, C_out=$C_out, H=$H")
        println("─"^60)
        
        # Get results for this configuration
        im_result = filter(r -> r.implementation == "im" && 
                              r.batch_size == N && 
                              r.in_channels == C_in && 
                              r.out_channels == C_out && 
                              r.spatial_size == H, 
                         results)[1]
        col_result = filter(r -> r.implementation == "col" && 
                               r.batch_size == N && 
                               r.in_channels == C_in && 
                               r.out_channels == C_out && 
                               r.spatial_size == H, 
                          results)[1]
        
        print_comparison(im_result, col_result)
    end
end

# Run the benchmarks
function run_full_benchmark()
    println("Starting comprehensive convolution benchmarks...")
    results = benchmark_convolutions(
        batch_sizes=[16, 32],
        in_channels=[32, 64],
        out_channels=[32, 64],
        spatial_sizes=[28, 56],
        warmup=3,
        iterations=10
    )
    
    summarize_results(results)
    return results
end