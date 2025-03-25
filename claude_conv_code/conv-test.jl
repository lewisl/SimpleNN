using LinearAlgebra
using Statistics
using Test

include("colconv.jl")

function test_conv_layer_numeric()
    # Test parameters
    batch_size, height, width = 2, 6, 6
    in_channels, out_channels = 2, 3
    kernel_size = 3
    stride = 1
    pad = 1
    
    # Initialize layer with specific values for reproducibility
    weights = ones(Float64, kernel_size, kernel_size, in_channels, out_channels)
    bias = zeros(Float64, out_channels)
    layer = ConvLayer(weights, bias, stride, pad)
    
    # Create input with specific pattern
    x = ones(Float64, height, width, in_channels, batch_size)
    
    # Forward pass
    conv_inputs, img_size = prep_forward(layer, x)
    out = forward_conv(conv_inputs...)
    
    #println("Checking convolution output values:")
    
    # Test middle values (should all be 18.0)
    #println("\nFull middle section (should all be 18.0):")
    #println(out[2:5, 2:5, 1, 1])  # Show first channel, first batch
    # display(out)
    
    middle_correct = true
    for c in 1:out_channels
        for b in 1:batch_size
            if !all(isapprox.(out[2:5, 2:5, c, b], 18.0, atol=1e-10))
                middle_correct = false
                #println("Mismatch in middle values at channel $c, batch $b:")
                #println(out[2:5, 2:5, c, b])
                break
            end
        end
    end
    @test middle_correct
    
    # Test edge values (should be 12.0)
    #println("\nEdge values (should be 12.0):")
    #println("Top edge: ", out[1, 2:5, 1, 1])  # top edge, first channel/batch
    
    edges_correct = true
    for c in 1:out_channels
        for b in 1:batch_size
            # Top and bottom edges (middle columns)
            if !all(isapprox.(out[1, 2:5, c, b], 12.0, atol=1e-10)) ||
               !all(isapprox.(out[6, 2:5, c, b], 12.0, atol=1e-10))
                edges_correct = false
                #println("Edge mismatch in channel $c, batch $b")
                #println("Top edge: ", out[1, 2:5, c, b])
                #println("Bottom edge: ", out[6, 2:5, c, b])
                break
            end
            
            # Left and right edges (middle rows)
            if !all(isapprox.(out[2:5, 1, c, b], 12.0, atol=1e-10)) ||
               !all(isapprox.(out[2:5, 6, c, b], 12.0, atol=1e-10))
                edges_correct = false
                #println("Edge mismatch in channel $c, batch $b")
                #println("Left edge: ", out[2:5, 1, c, b])
                #println("Right edge: ", out[2:5, 6, c, b])
                break
            end
        end
    end
    @test edges_correct
    
    # Test corner values (should be 8.0)
    #println("\nCorner values (should be 8.0):")
    #println("Corners from first channel, first batch:")
    #println("TL: ", out[1,1,1,1], " TR: ", out[1,6,1,1])
    #println("BL: ", out[6,1,1,1], " BR: ", out[6,6,1,1])
    
    corners_correct = true
    for c in 1:out_channels
        for b in 1:batch_size
            corners = [out[1,1,c,b], out[1,6,c,b],  # top corners
                      out[6,1,c,b], out[6,6,c,b]]   # bottom corners
            if !all(isapprox.(corners, 8.0, atol=1e-10))
                corners_correct = false
                #println("Corner mismatch in channel $c, batch $b:")
                #println("Values: ", corners)
                break
            end
        end
    end
    @test corners_correct
    
    return true
end


function test_im2col_col2im()
    # Test parameters
    batch_size, height, width = 2, 4, 4
    in_channels = 2
    kernel_size = 3
    stride = 1
    pad = 1
    
    # Create input
    x = randn(Float64, height, width, in_channels, batch_size)
    
    # Test im2col
    x_col = im2col(x, kernel_size, stride, pad)
    
    # Test dimensions
    exp_out_height = div((height + 2pad - kernel_size), stride) + 1
    exp_out_width = div((width + 2pad - kernel_size), stride) + 1
    exp_col_rows = kernel_size * kernel_size * in_channels
    exp_col_cols = batch_size * exp_out_height * exp_out_width
    
    @test size(x_col) == (exp_col_rows, exp_col_cols)
    
    # Test col2im
    x_shape = (height=height, width=width, channels=in_channels, batch_size=batch_size)
    x_reconstructed = col2im(x_col, x_shape, kernel_size, stride, pad)
    
    # Test reconstruction size
    @test size(x_reconstructed) == size(x)
    
    return true
end

function test_pool_layer()
    # Test parameters
    batch_size, height, width = 2, 6, 6
    channels = 2
    pool_size = 2
    stride = 2
    
    # Create input with known pattern
    x = zeros(Float64, height, width, channels, batch_size)
    # Set specific values in first channel, first batch
    x[1:2, 1:2, 1, 1] = [1.0 2.0; 3.0 4.0]
    x[3:4, 1:2, 1, 1] = [5.0 6.0; 7.0 8.0]
    
    # Test max pooling
    max_layer = PoolLayer(pool_size, stride, :max)
    max_out, max_locs = forward_pool(x, max_layer)
    
    # Expected output size
    exp_height = div(height - pool_size, stride) + 1
    exp_width = div(width - pool_size, stride) + 1
    
    # Test dimensions
    @test size(max_out) == (exp_height, exp_width, channels, batch_size)
    
    # Test max pooling values
    @test max_out[1,1,1,1] == 4.0  # Max of first 2x2 block
    @test max_out[2,1,1,1] == 8.0  # Max of second 2x2 block
    
    # Test mean pooling
    mean_layer = PoolLayer(pool_size, stride, :mean)
    mean_out, mean_locs = forward_pool(x, mean_layer)
    
    # Test mean pooling values
    @test mean_out[1,1,1,1] ≈ 2.5  # Mean of first 2x2 block
    @test mean_out[2,1,1,1] ≈ 6.5  # Mean of second 2x2 block
    
    # Test backprop for max pooling
    dL_dout = ones(Float64, size(max_out))
    dL_dx_max = backward_pool(max_layer, dL_dout, size(x), max_locs)
    
    # Test backprop dimensions
    @test size(dL_dx_max) == size(x)
    
    # Test gradient flow - should be 1.0 at max locations, 0.0 elsewhere
    @test dL_dx_max[2,2,1,1] == 1.0  # Location of max in first block
    @test dL_dx_max[4,2,1,1] == 1.0  # Location of max in second block
    
    # Test backprop for mean pooling
    dL_dx_mean = backward_pool(mean_layer, dL_dout, size(x), mean_locs)
    # @show dL_dx_mean

    # Test mean pooling gradient - should be 1/4 everywhere in the window
    @test all(dL_dx_mean[1:2,1:2,1,1] .≈ 1.0)
    
    return true
end

function test_backprop_numeric()
    # Test parameters
    batch_size, height, width = 2, 6, 6
    in_channels, out_channels = 2, 3
    kernel_size = 3
    stride = 1
    pad = 1
    
    # Initialize layer with specific weights for testing
    weights = ones(Float64, kernel_size, kernel_size, in_channels, out_channels)
    bias = zeros(Float64, out_channels)
    layer = ConvLayer(weights, bias, stride, pad)
    
    # Create input
    x = ones(Float64, height, width, in_channels, batch_size)
    
    # Forward pass
    conv_inputs, img_size = prep_forward(layer, x)
    out = forward_conv(conv_inputs...)
    
    # Create gradient with known pattern
    dL_dout = ones(Float64, size(out))
    
    # Backward pass
    dL_dx, dL_dw, dL_db = backward_conv(layer, dL_dout, img_size, conv_inputs)
    
    # Test gradient dimensions
    @test size(dL_dx) == size(x)
    @test size(dL_dw) == size(weights)
    @test size(dL_db) == size(bias)
    
    # Test bias gradient
    # Each output pixel contributes 1.0 to the bias gradient
    expected_bias_grad = fill(height * width * batch_size, out_channels)
    @test all(isapprox.(dL_db, expected_bias_grad))
    
    # Test weight gradient properties
    # With all-ones input and all-ones upstream gradient:
    # Each weight should receive gradient contributions from all positions
    @test minimum(dL_dw) > 0  # All gradients should be positive
    @test all(dL_dw[:,:,1,1] .≈ dL_dw[:,:,1,2])  # Should be same for all output channels
    
    return true
end

# Run all tests
@testset verbose=true "Neural Network Layer Tests" begin
    @testset "Convolution Layer Numeric Tests" begin
        @test test_conv_layer_numeric()
    end

    @testset "Convolutional Layer im2col and col2im Tests" begin
        @test test_im2col_col2im()
    end
    
    @testset "Pooling Layer Tests" begin
        @test test_pool_layer()
    end
    
    @testset "Backpropagation Numeric Tests" begin
        @test test_backprop_numeric()
    end
end
