using SimpleNN
using Test
using BenchmarkTools
using LinearAlgebra
using Statistics
using Random

Random.seed!(1234)
const ELT = Float32

@testset "Layer Evaluation" begin

    @testset "ConvLayer" begin
        println("\n=== Testing ConvLayer ===")
        batch_size = 2
        # Setup: Input -> Conv1 -> Conv2
        # We test Conv1. Conv2 acts as the "layer above" for backprop.

        l1_spec = inputlayerspec(name=:input, h=5, w=5, outch=2)
        l1 = InputLayer(l1_spec, batch_size)
        rand!(l1.a)

        l2_spec = convlayerspec(name=:conv1, outch=3, f_h=3, f_w=3, activation=:relu, padrule=:same)
        l2 = ConvLayer(l2_spec, l1, batch_size)

        l3_spec = convlayerspec(name=:conv2, outch=2, f_h=3, f_w=3, activation=:relu, padrule=:same)
        l3 = ConvLayer(l3_spec, l2, batch_size)

        # Forward
        l2(l1.a, batch_size)
        @test size(l2.a) == (5, 5, 3, batch_size)
        println("  Forward Pass Dimensions: OK")

        # Backward
        # Initialize l3.eps_l to simulate incoming error
        rand!(l3.eps_l)

        # Run backward on l2
        l2(l3, batch_size)

        @test !all(iszero, l2.grad_weight)
        @test !all(iszero, l2.grad_bias)
        @test !all(iszero, l2.eps_l)
        println("  Backward Pass Gradients: OK")

        # Benchmark
        println("Benchmarking ConvLayer...")
        b_fwd = @benchmark $l2($l1.a, $batch_size)
        println("  Forward: ", median(b_fwd))
        b_bwd = @benchmark $l2($l3, $batch_size)
        println("  Backward: ", median(b_bwd))
    end

    @testset "MaxPoolLayer" begin
        println("\n=== Testing MaxPoolLayer ===")
        batch_size = 2
        # Setup: Input -> Conv -> MaxPool -> Conv
        # MaxPool requires prev layer to have grad_a (so we use Conv)

        l1_spec = inputlayerspec(name=:input, h=4, w=4, outch=2)
        l1 = InputLayer(l1_spec, batch_size)
        rand!(l1.a)

        l2_spec = convlayerspec(name=:conv_pre, outch=2, f_h=1, f_w=1, activation=:none, padrule=:same)
        l2 = ConvLayer(l2_spec, l1, batch_size)
        # Manually set l2.a to known values for testing max selection
        l2.a .= reshape(1:length(l2.a), size(l2.a))

        l3_spec = maxpoollayerspec(name=:pool, f_h=2, f_w=2)
        l3 = MaxPoolLayer(l3_spec, l2, batch_size)

        l4_spec = convlayerspec(name=:conv_post, outch=2, f_h=1, f_w=1, padrule=:same)
        l4 = ConvLayer(l4_spec, l3, batch_size)

        # Forward
        l3(l2.a, batch_size)
        @test size(l3.a) == (2, 2, 2, batch_size)

        # Verify max value
        # l2.a is 4x4x2x2
        # First block 2x2 of first channel, first batch
        val_1_1 = l2.a[1, 1, 1, 1]
        val_2_1 = l2.a[2, 1, 1, 1]
        val_1_2 = l2.a[1, 2, 1, 1]
        val_2_2 = l2.a[2, 2, 1, 1]
        max_val = max(val_1_1, val_2_1, val_1_2, val_2_2)
        @test l3.a[1, 1, 1, 1] == max_val
        println("  Forward Pass Logic: OK")

        # Backward
        rand!(l4.eps_l)
        l3(l4, batch_size)

        # Check eps_l propagation
        grad_block = l3.eps_l[1:2, 1:2, 1, 1]
        @test count(!iszero, grad_block) == 1
        @test grad_block[argmax(grad_block)] == l4.eps_l[1, 1, 1, 1]
        println("  Backward Pass Logic: OK")

        # Benchmark
        println("Benchmarking MaxPoolLayer...")
        b_fwd = @benchmark $l3($l2.a, $batch_size)
        println("  Forward: ", median(b_fwd))
        b_bwd = @benchmark $l3($l4, $batch_size)
        println("  Backward: ", median(b_bwd))
    end

    @testset "FlattenLayer" begin
        println("\n=== Testing FlattenLayer ===")
        batch_size = 2
        h, w, c = 3, 3, 2

        l1_spec = inputlayerspec(name=:input, h=h, w=w, outch=c)
        l1 = InputLayer(l1_spec, batch_size)
        rand!(l1.a)

        l2_spec = flattenlayerspec(name=:flat)
        l2 = FlattenLayer(l2_spec, l1, batch_size)

        l3_spec = linearlayerspec(name=:linear, outputdim=5)
        l3 = LinearLayer(l3_spec, l2, batch_size)

        # Forward
        l2(l1.a, batch_size)
        @test size(l2.a) == (h * w * c, batch_size)
        println("  Forward Pass Dimensions: OK")

        # Backward
        rand!(l3.eps_l)
        rand!(l3.weight)
        l2(l3, batch_size)

        @test size(l2.eps_l) == (h, w, c, batch_size)
        println("  Backward Pass Dimensions: OK")

        # Benchmark
        println("Benchmarking FlattenLayer...")
        b_fwd = @benchmark $l2($l1.a, $batch_size)
        println("  Forward: ", median(b_fwd))
        b_bwd = @benchmark $l2($l3, $batch_size)
        println("  Backward: ", median(b_bwd))
    end

    @testset "LinearLayer" begin
        println("\n=== Testing LinearLayer ===")
        batch_size = 2
        in_dim = 10
        out_dim = 5

        l1_spec = inputlayerspec(name=:input, outputdim=in_dim)
        l1 = InputLayer(l1_spec, batch_size)
        rand!(l1.a)

        l2_spec = linearlayerspec(name=:hidden, outputdim=out_dim, activation=:relu)
        l2 = LinearLayer(l2_spec, l1, batch_size)

        l3_spec = outputlayerspec(name=:output, outputdim=3, activation=:softmax)
        l3 = LinearLayer(l3_spec, l2, batch_size)

        # Forward
        l2(l1.a, batch_size)
        @test size(l2.a) == (out_dim, batch_size)
        println("  Forward Pass Dimensions: OK")

        # Backward
        rand!(l3.eps_l)
        rand!(l3.weight)
        l2(l3, batch_size)

        @test !all(iszero, l2.grad_weight)
        @test !all(iszero, l2.grad_bias)
        @test !all(iszero, l2.eps_l)
        println("  Backward Pass Gradients: OK")

        # Benchmark
        println("Benchmarking LinearLayer...")
        b_fwd = @benchmark $l2($l1.a, $batch_size)
        println("  Forward: ", median(b_fwd))
        b_bwd = @benchmark $l2($l3, $batch_size)
        println("  Backward: ", median(b_bwd))
    end

end
