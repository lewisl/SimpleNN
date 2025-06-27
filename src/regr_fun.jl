using Distributions

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
        Y = (slope * X) .+ b .+ y_err'
    elseif length(slope) > 1   
        Y = (slope' * X) .+ b .+ y_err'
    end
    return X, Y
end