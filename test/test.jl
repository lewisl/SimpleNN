# ============================
# Functions for performance and memory testing
# ============================
using BenchmarkTools
using LinearAlgebra
using SimpleNN

const ELT = Float32 
layers = []


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

function softmax!(a::Array{ELT,2}, z::Array{ELT,2})
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


for (i, layer) in enumerate(layers)
    println("layer ", i, " ",typeof(layer))
    if (typeof(layer) <: ConvLayer) | (typeof(layer) <: LinearLayer)
        ok = layer.activationf === RELU_PTR
        println("layer $i : ", ok ? "✓" : "✗  (pointer mismatch)")
        if !ok
            @show i typeof(layer) objectid(layer.activationf) objectid(RELU_PTR)
        end
    end
end


# padding and image size calculations

# TODO do we want ceiling or div?
dim_out(imgx, filx, stride, pad) = div(imgx - filx + 2pad, stride) + 1
same_pad(imgx,filx,stride) = div((imgx * (stride - 1) - stride + filx), 2)

tst_img = [  
    #imgdim, fildim, pad, stride
    (28, 3, 1, 1)
    (28, 5, 1, 0)  
    (28, 5, 1, 1)
    (28, 5, 1, 2)
    (27,3,1,1)
    (27,5,1,0)
    (27,5,1,1)
    (27,5,1,2)
]


function train!( x, y, full_batch, epochs, minibatch_size=0)   #  where {L<:Layer}   layers::Vector{L};  , hp=default_hp

    dobatch = if minibatch_size == 0
                false
            elseif minibatch_size <= 39
                error("Minibatch_size too small.  Choose a larger minibatch_size.")
            elseif full_batch / minibatch_size > 3
                true
            else
                error("Minibatch_size too large with fewer than 3 batches. Choose a much smaller minibatch_size.")
            end

    # @show n_minibatches
    # @show n_samples

    # stats = allocate_stats(full_batch, minibatch_size, epochs)
    batch_counter = 0

    for e = 1:epochs
        println("epoch: ", e)


        loop = true
        samples_left = full_batch
        start_obs = end_obs = 0
        @inbounds while loop 

            if dobatch
                if samples_left > minibatch_size
                    start_obs = end_obs + 1
                    end_obs = start_obs + minibatch_size - 1
                else
                    start_obs = end_obs + 1
                    end_obs = start_obs + samples_left - 1
                    loop = false
                end
                x_part = view(x, ndims(x), start_obs:end_obs)
                y_part = view(y, ndims(y), start_obs:end_obs)
                samples_left -= minibatch_size
            else
                x_part = x
                y_part = y
                loop = false
            end

            batch_counter += 1

            println("    counter = ", batch_counter, " batch size: ", end_obs - start_obs + 1, " start_obs: ", start_obs, " end_obs: ", end_obs, " samples_left: ",samples_left)
        

            # feedforward!(layers, x_part)

            # backprop!(layers, y_part)

            # update_weight_loop!(layers, hp, batch_counter)

            # hp.do_stats && gather_stats!(stats, layers, y_part, batch_counter, batno, e; to_console=false)

        end
    end

    return # stats
end