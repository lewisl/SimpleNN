# %%

using LazyTables
using CSV
using SimpleNN
using StatsBase

const ELT = Float32

function load_data(df)
    tmp = (LazyTable(CSV.File(df)))
    tmp = filter(row -> row.horsepower != "?", tmp)
end




# %%   load data

cars = load_data("data/auto_mpg.csv")



# %%

y = Float32.(cars.mpg)';
x = hcat(Float32.(cars.cylinders), Float32.(cars.displacement), 
        parse.(Float32,cars.horsepower), Float32.(cars.weight), 
        Float32.(cars.acceleration))';
@show size(y)
@show size(x)

features = size(x,1)
x_std = standardize_features(x)

# %%   standardize debugging

# Add after standardization:
# println("Original x stats:")
# for i in 1:size(x,1)
#     println("Feature $i: mean=$(round(mean(x[i,:]), digits=3)), std=$(round(std(x[i,:]), digits=3))")
# end

# println("\nStandardized x_std stats:")
# for i in 1:size(x_std,1)
#     println("Feature $i: mean=$(round(mean(x_std[i,:]), digits=3)), std=$(round(std(x_std[i,:]), digits=3))")
# end

# println("\nTarget y stats:")
# println("y: mean=$(round(mean(y), digits=3)), std=$(round(std(y), digits=3)), min=$(round(minimum(y), digits=3)), max=$(round(maximum(y), digits=3))")

# %%      using Julia built-in linear regression
  

# Quick comparison with Julia's built-in linear regression
# using LinearAlgebra

# Convert to Float64 for compatibility and add bias column
# X_with_bias = [ones(Float64, size(x_std, 2)) Float64.(x_std')]  # Add intercept column
# y_float64 = Float64.(vec(y))

# Solve normal equations: β = (X'X)^(-1)X'y
# beta = (X_with_bias' * X_with_bias) \ (X_with_bias' * y_float64)

# Make predictions
# lm_pred = X_with_bias * beta
# lm_error = mean(abs.(y_float64 .- lm_pred) ./ y_float64) * 100

# println("Linear Regression (Normal Equations) Error: $(round(lm_error, digits=2))%")
# println("Neural Network Error: 9.7%")
# println("Difference: $(round(abs(lm_error - 9.7), digits=2))%")

# %%

regr_model = LayerSpec[
    inputlayerspec(outputdim=features, name=:input)
    outputlayerspec(outputdim=1, activation=:regression, name=:output)
];
preptest = false
fullbatch = size(y,2)
minibatch_size = fullbatch
epochs = 1000
layerspecs = regr_model

hp = HyperParameters(lr=ELT(0.008), reg=:none, regparm=ELT(0.00013), do_stats=false)  # reg=:L2, regparm=0.00043,

# %%  # setup the layers: set array sizes and pre-allocate data and weight arrays

layers = setup_train(layerspecs, minibatch_size);
# @show layers[end].weight



# %%  train the model

stats = train!(layers; x=x_std, y=y, fullbatch=fullbatch,
    epochs=epochs, minibatch_size=minibatch_size, hp=hp);
