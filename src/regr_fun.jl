using Distributions
using LinearAlgebra
using Statistics
using Random

# some sample data
    x_mod = [(1.0,0.2), (3.0,0.4),(7.0, 1.0)] # tuple of tuple of mean, std
    slope = [1.0, 2.0, 3.0]
    b = 0.4

"""
    lr_data(x_spec, n, slope, b, y_std=1.0)

Create sample data for linear regression

# Arguments
- `x_spec`: a vector of tuples; each tuple is the mean and std. dev of an independent variable in X
- `n`: the number of data points as an integer
- `slope`: a vector of slopes for each feature in x; the length of slope must match the length of x_spec
- `b`: the intercept
- `y_std`: the standard deviation for the error in Y
"""
function lr_data(x_spec, n, slope, b, y_std=1.0)
    @assert length(x_spec) == length(slope)
    X = reduce(vcat, [rand(Normal(x[1], x[2]),n) for x in x_spec]')
    y_err = rand(Normal(0.0, y_std), n)
    if length(slope) == 1
        Y = (slope .* X) .+ b .+ y_err'
    elseif length(slope) > 1
        Y = (slope' * X) .+ b .+ y_err'
    end
    return X, Y
end


"""
    calculate_r2(y_pred, y_true)

Calculate the coefficient of determination (R²) for regression results.

# Arguments
- `y_true`: actual target values
- `y_pred`: predicted values from the model

# Returns
- R² value (coefficient of determination)
"""
function calculate_r2(y_pred, y_true)
    y_true_v = view(y_true,:) # Convert to vector view--no copying or extra allocation
    y_pred_v = view(y_pred,:)

    # Calculate means
    y_mean = mean(y_true_v)

    # Calculate sum of squares
    ss_tot = sum((y_true_v .- y_mean).^2)  # Total sum of squares
    ss_res = sum((y_true_v .- y_pred_v).^2)  # Residual sum of squares

    # Calculate R²
    r2 = 1.0 - (ss_res / ss_tot)

    return r2
end


"""
    partial_r_squared(X, y, weights, bias, feature_idx)

Calculate the partial R-squared (coefficient of partial determination) for a specific predictor.

Partial R-squared represents the proportion of remaining variance in the dependent
variable that is explained by a specific predictor, after accounting for all other
predictors in the model.

Formula: R²[Y,X₂|X₁] = (SSE(reduced) - SSE(full)) / SSE(reduced)

This answers: "What percentage of the variance not already explained by other
variables can this variable explain?"

# Arguments
- `X`: input features matrix (features × samples)
- `y`: target values
- `weights`: regression coefficients (from neural network output layer)
- `bias`: bias term (from neural network output layer)
- `feature_idx`: index of the feature to analyze (1-based)

# Returns
- Partial R² value (proportion of remaining variance explained)
"""
function partial_r_squared(X, y, weights, bias, feature_idx)
    y_v = view(y, :)

    # Full model prediction
    y_pred_full = weights * X .+ bias
    y_pred_full_v = view(y_pred_full, :)
    sse_full = sum((y_v .- y_pred_full_v).^2)

    # Reduced model: remove the specified feature (set coefficient to 0)
    weights_reduced = copy(weights)
    weights_reduced[1, feature_idx] = 0.0
    y_pred_reduced = weights_reduced * X .+ bias
    y_pred_reduced_v = view(y_pred_reduced, :)
    sse_reduced = sum((y_v .- y_pred_reduced_v).^2)

    # Partial R² = (SSE_reduced - SSE_full) / SSE_reduced
    partial_r2 = (sse_reduced - sse_full) / sse_reduced

    return partial_r2
end

"""
    semi_partial_r_squared(X, y, weights, bias, feature_idx)

Calculate the semi-partial (part) correlation squared for a specific predictor.

Semi-partial correlation squared represents the amount by which R² would decrease
if that variable were removed from the model. This measures the unique contribution
of each variable to the overall R².

Formula: sr²ᵢ = R²(full) - R²(without variable i)

This is the most direct measure of unique variance contribution and these values
can be summed to understand total unique contributions.

# Arguments
- `X`: input features matrix (features × samples)
- `y`: target values
- `weights`: regression coefficients (from neural network output layer)
- `bias`: bias term (from neural network output layer)
- `feature_idx`: index of the feature to analyze (1-based)

# Returns
- Semi-partial R² value (unique contribution to overall R²)
"""
function semi_partial_r_squared(X, y, weights, bias, feature_idx)
    # Full model R²
    y_pred_full = weights * X .+ bias
    r2_full = calculate_r2(y_pred_full, y)

    # Reduced model R²: remove the specified feature (set coefficient to 0.0)
    weights_reduced = copy(weights)
    weights_reduced[1, feature_idx] = 0.0
    y_pred_reduced = weights_reduced * X .+ bias
    r2_reduced = calculate_r2(y_pred_reduced, y)

    # Semi-partial R² = R²_full - R²_reduced
    semi_partial_r2 = r2_full - r2_reduced

    return semi_partial_r2
end


"""
    analyze_regression_variance(X, y, output_layer)

Perform comprehensive variance analysis using partial and semi-partial R² methods.

This function calculates how much each predictor variable contributes to explaining
the variance in the dependent variable. It computes both partial R² (proportion of
remaining variance explained by each predictor) and semi-partial R² (unique
contribution to overall model R²).

# Arguments
- `X`: input features matrix (features × samples)
- `y`: target values
- `output_layer`: the final layer of the neural network (contains weights and bias)

# Returns
- Dictionary with overall R² and variance contributions from partial and semi-partial R² methods
"""
function analyze_regression_variance(X, y, output_layer)
    # Extract weights and bias from output layer
    weights = output_layer.weight
    bias = output_layer.bias

    # Calculate overall R²
    y_pred = weights * X .+ bias
    overall_r2 = calculate_r2(y_pred, y)

    # Calculate variance contributions using both methods
    n_features = size(weights, 2)  # weights is (output_dim, input_dim) = (1, n_features)
    partial_r2 = Vector{Float64}(undef, n_features)
    semi_partial_r2 = Vector{Float64}(undef, n_features)

    for i in 1:n_features
        partial_r2[i] = partial_r_squared(X, y, weights, bias, i)
        semi_partial_r2[i] = semi_partial_r_squared(X, y, weights, bias, i)
    end

    # Create results dictionary
    results = Dict(
        "overall_r2" => overall_r2,
        "partial_r2" => partial_r2,
        "semi_partial_r2" => semi_partial_r2,
        "coefficients" => view(weights, :),
        "bias" => bias,
        "n_features" => n_features
    )

    return results
end

"""
    print_variance_analysis(results; feature_names=nothing)

Print a formatted report of the variance analysis results.

This function displays a comprehensive breakdown of how each predictor variable
contributes to explaining variance in the dependent variable. It shows both
partial R² values (proportion of remaining variance explained) and semi-partial
R² values (unique contribution to overall model R²).

# Arguments
- `results`: output from analyze_regression_variance()
- `feature_names`: optional vector of feature names for display
"""
function print_variance_analysis(results; feature_names=nothing)
    println("=== Regression Variance Analysis ===")
    println()

    # Overall R²
    println("Overall R²: $(round(results["overall_r2"], digits=4))")
    println("Overall variance explained: $(round(results["overall_r2"] * 100, digits=2))%")
    println()

    # Individual coefficients
    println("Variance contributions by feature:")
    println("----------------------------------")

    n_features = results["n_features"]

    for i in 1:n_features
        feature_name = feature_names !== nothing ? feature_names[i] : "Feature $i"
        coeff_val = results["coefficients"][i]
        # partial_r2 = results["partial_r2"][i]
        # semi_partial_r2 = results["semi_partial_r2"][i]

        println("$feature_name:")
        println("  Coefficient: $(round(coeff_val, digits=4))")
        # println("  Partial R²: $(round(partial_r2, digits=4)) ($(round(partial_r2 * 100, digits=2))%)")
        # println("    → Of remaining unexplained variance, this feature explains $(round(partial_r2 * 100, digits=1))%")
        # println("  Semi-Partial R²: $(round(semi_partial_r2, digits=4)) ($(round(semi_partial_r2 * 100, digits=2))%)")
        println("    → Unique contribution to overall R²")
        println()
    end

    # Bias
    bias_val = length(results["bias"]) == 1 ? results["bias"][1] : results["bias"]
    println("Bias (intercept): $(round(bias_val, digits=4))")

    # Show the relationship between methods
    total_semi_partial_r2 = sum(results["semi_partial_r2"])
    println()
    println("Method Comparison:")
    println("Sum of Semi-Partial R²: $(round(total_semi_partial_r2, digits=4))")
    println("Overall model R²: $(round(results["overall_r2"], digits=4))")
    println()
    println("Interpretation:")
    println("• Partial R²: What % of remaining variance each feature explains")
    println("• Semi-Partial R²: Unique contribution to overall R² (these sum up)")
    println("• Semi-partial values sum to overall R² when features are uncorrelated")
end

# Test layer structure for testing functions
struct TestLayer
    weight::Matrix{Float64}
    bias::Vector{Float64}
end


"""
    test_variance_functions()

Run comprehensive tests of all variance analysis functions with known data.
Returns true if all tests pass, false otherwise.
"""
function test_variance_functions()
    println("Running variance analysis function tests...")

    # Test 1: Perfect linear relationship
    println("Test 1: Perfect linear relationship")
    X_perfect = reshape([1.0, 2.0, 3.0, 4.0, 5.0], 1, :)
    y_perfect = 2.0 * X_perfect[1, :] .+ 1.0  # y = 2x + 1

    layer_perfect = TestLayer(reshape([2.0], 1, 1), [1.0])

    results_perfect = analyze_regression_variance(X_perfect, y_perfect, layer_perfect)

    if abs(results_perfect["overall_r2"] - 1.0) < 1e-10
        println("✓ Perfect correlation test passed")
    else
        println("✗ Perfect correlation test failed: R² = $(results_perfect["overall_r2"])")
        return false
    end

    # Test 2: Multiple features with known contributions
    println("Test 2: Multiple features with known variance contributions")

    # Create data where we can predict variance contributions
    n = 1000
    Random.seed!(123)

    # Feature 1: high variance, low coefficient
    x1 = 3.0 * randn(n)  # variance ≈ 9
    # Feature 2: low variance, high coefficient
    x2 = 0.5 * randn(n)  # variance ≈ 0.25

    X_test = [x1'; x2']
    coeffs_test = [0.5, 4.0]  # Low coeff for high var feature, high coeff for low var feature
    bias_test = 2.0

    y_test = coeffs_test' * X_test .+ bias_test .+ 0.1 * randn(n)'

    layer_test = TestLayer(reshape(coeffs_test, :, 1), [bias_test])
    results_test = analyze_regression_variance(X_test, y_test, layer_test)

    # Feature 2 should explain more variance despite lower feature variance
    # because coefficient² × variance matters: (4.0)² × 0.25 = 4.0 vs (0.5)² × 9 = 2.25
    if results_test["variance_explained_pct"][2] > results_test["variance_explained_pct"][1]
        println("✓ Multiple features variance attribution test passed")
    else
        println("✗ Multiple features test failed")
        println("  Feature 1 variance explained: $(results_test["variance_explained_pct"][1])%")
        println("  Feature 2 variance explained: $(results_test["variance_explained_pct"][2])%")
        return false
    end

    # Test 3: R² calculation edge cases
    println("Test 3: R² calculation edge cases")

    # Constant target values
    y_constant = fill(5.0, 10)
    y_pred_constant = fill(5.0, 10)
    r2_constant = calculate_r2(y_pred_constant, y_constant)

    # Should handle this gracefully (either NaN or 1.0 is acceptable)
    if isnan(r2_constant) || abs(r2_constant - 1.0) < 1e-10
        println("✓ Constant values R² test passed")
    else
        println("✗ Constant values R² test failed: R² = $r2_constant")
        return false
    end

    println("All variance analysis tests passed! ✓")
    return true
end


"""
    compare_with_analytical_solution(X, y, output_layer)

Compare neural network results with analytical least squares solution.
Useful for validating that the neural network converged to the optimal solution.

# Arguments
- `X`: input features matrix (features × samples)
- `y`: target values
- `output_layer`: trained neural network output layer

# Returns
- Dictionary comparing neural network vs analytical solutions
"""
function compare_with_analytical_solution(X, y, output_layer)
    # Neural network solution
    nn_weights = view(output_layer.weight, :)
    nn_bias = output_layer.bias[1]

    # Analytical least squares solution: β = (X'X)⁻¹X'y
    # Add bias column to X for analytical solution
    X_with_bias = [X; ones(1, size(X, 2))]  # Add row of ones for bias

    # Solve normal equations
    analytical_params = (X_with_bias * X_with_bias') \ (X_with_bias * view(y, :))
    analytical_weights = analytical_params[1:end-1]
    analytical_bias = analytical_params[end]

    # Calculate predictions for both
    nn_pred = nn_weights' * X .+ nn_bias
    analytical_pred = analytical_weights' * X .+ analytical_bias

    # Calculate R² for both
    nn_r2 = calculate_r2(nn_pred, y)
    analytical_r2 = calculate_r2(analytical_pred, y)

    # Calculate differences
    weight_diff = norm(nn_weights - analytical_weights)
    bias_diff = abs(nn_bias - analytical_bias)
    r2_diff = abs(nn_r2 - analytical_r2)

    results = Dict(
        "nn_weights" => nn_weights,
        "analytical_weights" => analytical_weights,
        "nn_bias" => nn_bias,
        "analytical_bias" => analytical_bias,
        "nn_r2" => nn_r2,
        "analytical_r2" => analytical_r2,
        "weight_difference_norm" => weight_diff,
        "bias_difference" => bias_diff,
        "r2_difference" => r2_diff,
        "converged_well" => (weight_diff < 0.1 && bias_diff < 0.1 && r2_diff < 0.01)
    )

    return results
end

"""
    print_convergence_analysis(comparison_results)

Print a formatted report comparing neural network vs analytical solutions.
"""
function print_convergence_analysis(comparison_results)
    println("=== Neural Network vs Analytical Solution Comparison ===")
    println()

    println("Weights:")
    println("  Neural Network: $(round.(comparison_results["nn_weights"], digits=4))")
    println("  Analytical:     $(round.(comparison_results["analytical_weights"], digits=4))")
    println("  Difference (norm): $(round(comparison_results["weight_difference_norm"], digits=6))")
    println()

    println("Bias:")
    println("  Neural Network: $(round(comparison_results["nn_bias"], digits=4))")
    println("  Analytical:     $(round(comparison_results["analytical_bias"], digits=4))")
    println("  Difference:     $(round(comparison_results["bias_difference"], digits=6))")
    println()

    println("R² Values:")
    println("  Neural Network: $(round(comparison_results["nn_r2"], digits=6))")
    println("  Analytical:     $(round(comparison_results["analytical_r2"], digits=6))")
    println("  Difference:     $(round(comparison_results["r2_difference"], digits=8))")
    println()

    if comparison_results["converged_well"]
        println("✓ Neural network converged well to analytical solution!")
    else
        println("⚠ Neural network may need more training or different hyperparameters")
    end
end