# ============================
# Functions for performance and memory testing
# ============================
using BenchmarkTools
using LinearAlgebra


function slicemean!(arr_norm::AbstractArray, arr::AbstractArray, mu::AbstractVector, stddev::AbstractVector)
    @inbounds @fastmath for (c,ch_z, ch_z_norm) in zip(1:size(arr,3), eachslice(arr,dims=3),eachslice(arr_norm,dims=3)) # enumerate(eachslice(arr, dims=3, drop=true))
        mu[c] = mean(ch_z)
        stddev[c] = std(ch_z, corrected=false)
        @. ch_z_norm = (ch_z - mu[c]) / (stddev[c] + 1e-12)
    end
    return  # (mu=mu, stddev=stddev, z_norm=arr_norm)
end

function fastmu!(mu, x; dim)
    cnt = length(x) / size(x,dim)

    for (i, sl) in enumerate(eachslice(x,dims=dim))
        mu[i] = sum(sl) / cnt
    end     
end

function flatloop(arr1)
    flatdim = size(arr1, 1) * size(arr1, 2) * size(arr1, 3)
    ret = zeros(flatdim, size(arr1, 4))
    innerdim = 0
    outerdim = 0
    for b in axes(arr1, 4)
        outerdim += 1
        innerdim = 0
        for c in axes(arr1, 3)
            for j in axes(arr1, 2)
                for i in axes(arr1, 1)
                    innerdim += 1
                    ret[innerdim, outerdim] = arr1[i, j, c, b]
                end
            end
        end
    end
    return ret
end

function flatloop!(arrout, arrin)
    innerdim = 0
    outerdim = 0
    for b in axes(arrin, 4)
        outerdim += 1
        innerdim = 0
        for c in axes(arrin, 3)
            for j in axes(arrin, 2)
                for i in axes(arrin, 1)
                    innerdim += 1
                    arrout[innerdim, outerdim] = arrin[i, j, c, b]
                end
            end
        end
    end
    return
end

function flattenview!(arrout, arrin)
    @views begin
        # Flatten the first 3 dimensions of `x` into `layer.a`
        for idx in axes(arrin, 4)  # iterate over batch dimension (4th dimension)
            arrout[:, idx] .= arrin[:, :, :, idx][:]  # Flatten the first 3 dimensions and assign to `layer.a`
        end
    end
end

function softmax!(a::Array{Float64,2}, z::Array{Float64,2})
    for c in axes(z, 2) # columns = samples
        va = view(a, :, c)
        vz = view(z, :, c)
        va .= exp.(vz .- maximum(vz))
        va .= va ./ (sum(va) .+ 1e-12)
    end
    return
end

function test_plot(s, e)
    plot(s:e)
end

function loop_dot(arr1, arr2)
    sum = 0.0
    @inbounds for i in eachindex(arr1)
        sum += @views arr1[i] * arr2[i]
    end
    return sum
end

# comparing benchmarks for sum of element-wise array multiplication: in order from fastest to slowest
@benchmark sum(l * e for (l, e) in zip(local_patch, err)) (setup = (local_patch=fill(0.5,3,3); err=(fill(0.4,3,3))))
@benchmark mapreduce(splat(*), +, zip(err, local_patch)) (setup = (local_patch=fill(0.5,3,3); err=(fill(0.4,3,3))))
@benchmark dot(local_patch, err) (setup = (local_patch=fill(0.5,3,3); err=(fill(0.4,3,3))))
@benchmark sum(local_patch[i] * err[i] for i in eachindex(local_patch, err)) (setup = (local_patch=fill(0.5,3,3); err=(fill(0.4,3,3))))
@benchmark sum(local_patch .* err) (setup = (local_patch=fill(0.5,3,3); err=(fill(0.4,3,3))))
