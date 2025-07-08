
# ============================
# Utility Functions
# ============================

function random_onehot(i, j)
    arr = zeros(ELT, i, j)
    for n in axes(arr, 2)
        rowselector = rand(1:10)
        arr[rowselector, n] = ELT(1.0)
    end
    return arr
end

# Normalize features to [0,1] or standardize to mean=0, std=1
function normalize_features(x)
    x_norm = copy(x)
    for i in axes(x, 1)
        min_val = minimum(x[i, :])
        max_val = maximum(x[i, :])
        x_norm[i, :] = (x[i, :] .- min_val) ./ (max_val - min_val)
    end
    return x_norm
end

# Or standardize (often better for regression)
function standardize_features(x)
    x_std = copy(x)
    for i in axes(x,1)
        mu = mean(x[i, :])
        stddev = std(x[i, :])
        x_std[i, :] = (x[i, :] .- mu) ./ stddev
    end
    return x_std
end


# ============================
# Save and load weights to/from files
# ============================

function weights2file(layers, suffix, pathstr)
    for lr in fieldnames(typeof(layers))
        serialize(joinpath(pathstr, string(lr) * "_bias" * '_' * suffix * ".dat"),
            eval(getfield(layers, lr)).bias)
        serialize(joinpath(pathstr, string(lr) * "_weight" * '_' * suffix * ".dat"),
            eval(getfield(layers, lr)).weight)
    end
end

# TODO this can't work any more
# function file2weights(suffix, pathstr)
#     outlayers = init_layers(n_samples=batch_size)
#     for lr in fieldnames(typeof(outlayers))
#         fname_bias = joinpath(pathstr, string(lr) * "_bias" * '_' * suffix * ".dat")
#         fname_weight = joinpath(pathstr, string(lr) * "_weight" * '_' * suffix * ".dat")
#         setfield!(getfield(layers, lr), :bias, deserialize(fname_bias))
#         setfield!(getfield(layers, lr), :weight, deserialize(fname_weight))
#     end
#     return outlayers
# end
