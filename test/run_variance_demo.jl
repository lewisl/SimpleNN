# ==========================================================
# Demonstration script for variance analysis functions
# Generates data, trains neural network, and analyzes variance
# ==========================================================

# %% startup - required for non-Julia aware environment like Zed REPL

cd(joinpath(homedir(), "code", "SimpleNN"))
using Pkg
Pkg.activate(".")
Pkg.instantiate()

# %% packages and inputs

using SimpleNN
using Statistics
using Random

const ELT = Float32

# Include our variance analysis functions
include("regr_fun.jl")

println("=== Neural Network Variance Analysis Demonstration ===")
println()

# %% Test Case 1: Simple Linear Regression (1 feature)

println("TEST CASE 1: Simple Linear Regression (1 feature)")
println("=" ^ 50)

# Model definition for simple regression
simple_reg = LayerSpec[
    inputlayerspec(outputdim=1, name=:input)
    outputlayerspec(outputdim=1, activation=:regression, name=:output)
]

# Hyperparameters
hp_simple = HyperParameters(lr=ELT(0.01), reg=:L2, regparm=ELT(0.001), do_stats=false)

# Generate simple linear data: y = 2.5*x + 1.2 + noise
fullbatch_simple = 1000
minibatch_size_simple = 1000
epochs_simple = 500

xspec_simple = [(2.0, 0.8)]  # mean=2.0, std=0.8
slope_simple = [2.5]
bias_simple = 1.2

x_train_simple, y_train_simple = lr_data(xspec_simple, fullbatch_simple, slope_simple, bias_simple, 0.8)
x_train_simple = Float32.(x_train_simple)
y_train_simple = Float32.(y_train_simple)

println("True relationship: y = 2.5*x + 1.2 + noise")
println("Training samples: $fullbatch_simple")
println()

# Setup and train
layers_simple = setup_train(simple_reg, minibatch_size_simple)

println("Training simple regression model...")
train!(layers_simple; x=x_train_simple, y=y_train_simple, fullbatch=fullbatch_simple,
        epochs=epochs_simple, minibatch_size=minibatch_size_simple, hp=hp_simple)

# Get the output layer (last layer)
output_layer_simple = layers_simple[end]

println("Training completed!")
println("True coefficient: $slope_simple")
println("Learned weights: $(round.(vec(output_layer_simple.weight), digits=4))")
println("True bias: $bias_simple")
println("Learned bias: $(round.(output_layer_simple.bias, digits=4))")
println()

# Analyze variance
results_simple = analyze_regression_variance(x_train_simple, y_train_simple, output_layer_simple)
print_variance_analysis(results_simple, feature_names=["X_feature"])

println("\n" * "=" ^ 70 * "\n")

# %% Test Case 2: Multiple Linear Regression (3 features)

println("TEST CASE 2: Multiple Linear Regression (3 features)")
println("=" ^ 50)

# Model definition for multiple regression
multi_reg = LayerSpec[
    inputlayerspec(outputdim=3, name=:input)
    outputlayerspec(outputdim=1, activation=:regression, name=:output)
]

# Hyperparameters
hp_multi = HyperParameters(lr=ELT(0.005), reg=:L2, regparm=ELT(0.001), do_stats=false)

# Generate multiple regression data: y = 1.5*x1 + 3.0*x2 + 0.8*x3 + 2.0 + noise
fullbatch_multi = 2000
minibatch_size_multi = 200
epochs_multi = 800

# Different distributions for each feature to create varying importance
xspec_multi = [(1.0, 0.5), (0.0, 1.2), (-0.5, 2.0)]  # Different means and std devs
slope_multi = [1.5, 3.0, 0.8]  # Different coefficients
bias_multi = 2.0

x_train_multi, y_train_multi = lr_data(xspec_multi, fullbatch_multi, slope_multi, bias_multi, 1.0)
x_train_multi = Float32.(x_train_multi)
y_train_multi = Float32.(y_train_multi)

println("True relationship: y = 1.5*x1 + 3.0*x2 + 0.8*x3 + 2.0 + noise")
println("Feature 1: mean=1.0, std=0.5, coeff=1.5")
println("Feature 2: mean=0.0, std=1.2, coeff=3.0")  
println("Feature 3: mean=-0.5, std=2.0, coeff=0.8")
println("Training samples: $fullbatch_multi")
println()

# Setup and train
layers_multi = setup_train(multi_reg, minibatch_size_multi)

println("Training multiple regression model...")
train!(layers_multi; x=x_train_multi, y=y_train_multi, fullbatch=fullbatch_multi,
        epochs=epochs_multi, minibatch_size=minibatch_size_multi, hp=hp_multi)

# Get the output layer
output_layer_multi = layers_multi[end]

println("Training completed!")
println("True coefficients: $slope_multi")
println("Learned weights: $(round.(vec(output_layer_multi.weight), digits=4))")
println("True bias: $bias_multi")
println("Learned bias: $(round.(output_layer_multi.bias[1], digits=4))")
println()

# Analyze variance
results_multi = analyze_regression_variance(x_train_multi, y_train_multi, output_layer_multi)
feature_names = ["Temperature", "Pressure", "Humidity"]
print_variance_analysis(results_multi, feature_names=feature_names)

println("\n" * "=" ^ 70 * "\n")

# %% Test Case 3: Comparison with Known Statistical Results

println("TEST CASE 3: Validation Against Known Statistical Results")
println("=" ^ 50)

# Create a controlled example where we know the expected variance contributions
Random.seed!(42)  # For reproducibility

# Generate features with specific variances (no QR decomposition)
n_samples = 1500

# Create features with desired variances directly
X_ortho = zeros(Float64, 3, n_samples)
X_ortho[1, :] = randn(n_samples) * 1.0      # Feature 1: std=1.0, variance≈1.0
X_ortho[2, :] = randn(n_samples) * 2.0      # Feature 2: std=2.0, variance≈4.0
X_ortho[3, :] = randn(n_samples) * 0.5      # Feature 3: std=0.5, variance≈0.25

# Use equal coefficients so variance contribution depends only on feature variance
true_coeffs = [2.0, 2.0, 2.0]
true_bias_ortho = 1.0

# Add different noise levels for each feature to make it more realistic
feature_noise_levels = [0.6, 0.8, 0.4]  # Different noise for each feature
noise_components = zeros(n_samples)
for i in 1:3
    noise_components .+= feature_noise_levels[i] * randn(n_samples) * abs(true_coeffs[i])
end

y_ortho = true_coeffs' * X_ortho .+ true_bias_ortho .+ noise_components'

# Convert to Float32
X_ortho = Float32.(X_ortho)
y_ortho = Float32.(y_ortho)

println("Features with equal coefficients (2.0) but different variances and noise levels:")
println("Feature 1 variance: $(round(var(X_ortho[1, :]), digits=3)), noise level: $(feature_noise_levels[1])")
println("Feature 2 variance: $(round(var(X_ortho[2, :]), digits=3)), noise level: $(feature_noise_levels[2])")
println("Feature 3 variance: $(round(var(X_ortho[3, :]), digits=3)), noise level: $(feature_noise_levels[3])")
println("Expected: Feature 2 should explain most variance, Feature 3 least")
println()

# Train model
layers_ortho = setup_train(multi_reg, 200)
hp_ortho = HyperParameters(lr=ELT(0.001), reg=:L2, regparm=ELT(0.0001), do_stats=false)

println("Training orthogonal features model...")
train!(layers_ortho; x=X_ortho, y=y_ortho, fullbatch=n_samples,
        epochs=1000, minibatch_size=200, hp=hp_ortho)

output_layer_ortho = layers_ortho[end]

println("Training completed!")
println("True coefficients: $true_coeffs")
println("Learned weights: $(round.(vec(output_layer_ortho.weight), digits=4))")
println()

# Analyze variance
results_ortho = analyze_regression_variance(X_ortho, y_ortho, output_layer_ortho)
feature_names_ortho = ["Low_Var_Feature", "High_Var_Feature", "Very_Low_Var_Feature"]
print_variance_analysis(results_ortho, feature_names=feature_names_ortho)

println("\n" * "=" ^ 70 * "\n")

# %% Summary and Validation

println("SUMMARY AND VALIDATION")
println("=" ^ 30)

println("1. Simple Regression R²: $(round(results_simple["overall_r2"], digits=4))")
println("2. Multiple Regression R²: $(round(results_multi["overall_r2"], digits=4))")  
println("3. Orthogonal Features R²: $(round(results_ortho["overall_r2"], digits=4))")
println()

println("Key Insights:")
println("- Higher R² indicates better model fit")
println("- Variance explained percentages show feature importance")
println("- Features with higher variance × coefficient² contribute more")
println("- Sum of individual contributions may differ from overall R² due to interactions")

println("\nVariance analysis demonstration completed successfully!")
