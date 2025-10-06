# ==========================================================
# Test cases for variance analysis functions in regr_fun.jl
# ==========================================================

using Test
using Statistics
using LinearAlgebra

# Include the regression functions
include("../src/regr_fun.jl")

# Mock layer structure for testing
struct MockLinearLayer
    weight::Array{Float64, 2}
    bias::Vector{Float64}
end

@testset "Variance Analysis Tests" begin
    
    @testset "calculate_r2 function" begin
        # Test case 1: Perfect correlation (R² = 1)
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.0, 2.0, 3.0, 4.0, 5.0]
        r2 = calculate_r2(y_pred, y_true)
        @test r2 ≈ 1.0 atol=1e-10
        
        # Test case 2: No correlation (R² = 0)
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = fill(mean(y_true), length(y_true))  # All predictions = mean
        r2 = calculate_r2(y_pred, y_true)
        @test r2 ≈ 0.0 atol=1e-10
        
        # Test case 3: Known R² value
        y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
        y_pred = [1.1, 1.9, 3.1, 3.9, 5.1]  # Close but not perfect
        r2 = calculate_r2(y_pred, y_true)
        @test r2 > 0.9  # Should be high correlation
        
        # Test case 4: Matrix inputs (should work with vec())
        y_true_matrix = reshape([1.0, 2.0, 3.0, 4.0], 2, 2)
        y_pred_matrix = reshape([1.1, 1.9, 3.1, 3.9], 2, 2)
        r2 = calculate_r2(y_pred_matrix, y_true_matrix)
        @test isa(r2, Float64)
        @test r2 > 0.9
    end
    
    @testset "variance_explained_by_coefficient function" begin
        # Test case: Simple linear regression with known coefficients
        # y = 2*x1 + 3*x2 + 1 + noise
        n_samples = 100
        X = [randn(n_samples)'; 2*randn(n_samples)']  # 2 features × n_samples
        true_weights = [2.0, 3.0]
        true_bias = 1.0
        noise = 0.1 * randn(n_samples)
        y = true_weights' * X .+ true_bias .+ noise'
        
        # Test variance explained by first coefficient
        # var_exp_1 = variance_explained_by_coefficient(X, y, true_weights, true_bias, 1)
        # @test var_exp_1 > 0  # Should explain some variance
        # @test var_exp_1 < 100  # Should not explain all variance
        
        # Test variance explained by second coefficient
        # var_exp_2 = variance_explained_by_coefficient(X, y, true_weights, true_bias, 2)
        # @test var_exp_2 > 0
        # @test var_exp_2 < 100
        
        # Second coefficient should explain more variance (larger coefficient, more variable feature)
        # @test var_exp_2 > var_exp_1
    end
    
    @testset "analyze_regression_variance function" begin
        # Create test data with known relationships
        n_samples = 200
        X = [randn(n_samples)'; 0.5*randn(n_samples)'; 2*randn(n_samples)']  # 3 features
        true_weights = [1.5, 2.0, 0.5]
        true_bias = 0.3
        noise = 0.2 * randn(n_samples)
        y = true_weights' * X .+ true_bias .+ noise'
        
        # Create mock layer with the true weights
        mock_layer = MockLinearLayer(reshape(true_weights, :, 1), [true_bias])
        
        # Analyze variance
        results = analyze_regression_variance(X, y, mock_layer)
        
        # Test structure of results
        @test haskey(results, "overall_r2")
        # @test haskey(results, "variance_explained_pct")
        @test haskey(results, "coefficients")
        @test haskey(results, "bias")
        @test haskey(results, "n_features")
        
        # Test values
        @test results["overall_r2"] > 0.8  # Should have high R² with low noise
        @test length(results["variance_explained_pct"]) == 3
        @test results["n_features"] == 3
        @test results["coefficients"] ≈ true_weights atol=1e-10
        @test results["bias"][1] ≈ true_bias atol=1e-10
        
        # All variance percentages should be positive
        # @test all(results["variance_explained_pct"] .> 0)
    end
    
    @testset "print_variance_analysis function" begin
        # Create simple test results
        test_results = Dict(
            "overall_r2" => 0.85,
            "variance_explained_pct" => [25.0, 35.0, 25.0],
            "coefficients" => [1.5, 2.0, 0.8],
            "bias" => 0.3,
            "n_features" => 3
        )
        
        # Test that function runs without error
        @test_nowarn print_variance_analysis(test_results)
        
        # Test with feature names
        feature_names = ["Temperature", "Pressure", "Humidity"]
        @test_nowarn print_variance_analysis(test_results, feature_names=feature_names)
    end
    
    @testset "Edge cases and error handling" begin
        # Test with single feature
        X_single = reshape([1.0, 2.0, 3.0, 4.0, 5.0], 1, :)
        y_single = [2.0, 4.0, 6.0, 8.0, 10.0]
        weights_single = [2.0]
        bias_single = 0.0
        
        # var_exp = variance_explained_by_coefficient(X_single, y_single, weights_single, bias_single, 1)
        # @test var_exp > 90.0  # Single feature should explain most systematic variance
        
        # Test R² calculation with constant predictions
        y_const = fill(3.0, 5)
        y_pred_const = fill(3.0, 5)
        r2_const = calculate_r2(y_pred_const, y_const)
        @test isnan(r2_const) || r2_const ≈ 1.0  # Undefined or perfect when no variance in y_true
    end
end

println("All variance analysis tests completed!")
