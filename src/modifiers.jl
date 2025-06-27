
using LoopVectorization

# =====================
# normalization functions
# =====================

"""
struct Batch_norm_params holds batch normalization parameters for
feedfwd calculations and backprop training.
"""
Base.@kwdef struct BatchNorm{T<:AbstractArray} <: NormParam  # can this work for conv and linear???? array sizes differ
    # learned batch parameters to center and scale data
    gam::T  # scaling parameter for z_norm
    bet::T  # shifting parameter for z_norm (equivalent to bias)
    grad_gam::T
    grad_bet::T
    # for optimization updates of bn parameters
    grad_m_gam::T
    grad_v_gam::T
    grad_m_bet::T
    grad_v_bet::T
    # for standardizing batch values
    mu::T         # mean of z; same size as bias = no. of input layer units
    stddev::T       # std dev of z;   ditto
    mu_run::T      # running average of mu
    std_run::T       # running average of stddev
    istraining::Ref{Bool} = true     # set to false for inference or prediction
end

struct NoNorm <: NormParam
    @inline NoNorm() = new()   # help compiler elide any call to empty constructor
end   # a noop struct when not doing Batch Normalization


function batchnorm!(layer::LinearLayer, current_batch_size::Int)
    bn = layer.normparams
    cb = current_batch_size
    cb_rng = 1:cb
    vzn = view_minibatch(layer.z_norm, cb_rng)
    vz = view_minibatch(layer.z, cb_rng)

    if bn.istraining[]        # access value of Ref, like a 1 element array
        mean!(bn.mu, layer.z)
        bn.stddev .= std(layer.z, dims=2)

        @turbo @. @views vzn = (vz - bn.mu) / (bn.stddev + IT) # normalized: often xhat or zhat
        @turbo @. @views vz = vzn * bn.gam + bn.bet  # shift & scale: often called y
        @. bn.mu_run = ifelse(bn.mu_run[1] == ELT(0.0), bn.mu, ELT(0.95) * bn.mu_run + ELT(0.05) * bn.mu)
        @. bn.std_run = ifelse(bn.std_run[1] == ELT(0.0), bn.stddev, ELT(0.95) * bn.std_run + ELT(0.05) * bn.stddev)
    else  # prediction: use running mean and stddev
        @turbo @. vzn = (vz - bn.mu_run) / (bn.std_run + IT) # normalized: aka xhat or zhat
        @turbo @. vz = vzn * bn.gam + bn.bet  # shift & scale: often called y
    end
    return
end

function batchnorm!(layer::ConvLayer, current_batch_size)
    bn = layer.normparams
    cb = current_batch_size
    mb = size(layer.z, ndims(layer.z))
    cb_rng = 1:cb
    if mb == cb
        z_norm = layer.z_norm
        z = layer.z
    else
        z_norm = view_minibatch(layer.z_norm, cb_rng)
        z = view_minibatch(layer.z, cb_rng)
    end


    c = size(layer.z, 3)
    if bn.istraining[]    # access value of Ref, like a 1 element array
        @inbounds @fastmath for (cidx, ch_z, ch_z_norm) in zip(1:c, eachslice(z, dims=3), eachslice(z_norm, dims=3))
            # Compute statistics
            bn.mu[cidx] = mean(ch_z)
            bn.stddev[cidx] = std(ch_z, corrected=false)

            # Pre-compute inverse std for efficiency
            inv_std = ELT(1.0) / (bn.stddev[cidx] + IT)

            @turbo @. ch_z_norm = (ch_z - bn.mu[cidx]) * inv_std
            @turbo @. ch_z = ch_z_norm * bn.gam[cidx] + bn.bet[cidx]
        end

        # Update running statistics
        @. bn.mu_run = ifelse(bn.mu_run[1] == ELT(0.0), bn.mu, ELT(0.95) * bn.mu_run + ELT(0.05) * bn.mu)
        @. bn.std_run = ifelse(bn.std_run[1] == ELT(0.0), bn.stddev, ELT(0.95) * bn.std_run + ELT(0.05) * bn.stddev)
    else
        @inbounds @fastmath for (cidx, ch_z, ch_z_norm) in zip(1:c, eachslice(z, dims=3), eachslice(z_norm, dims=3))
            # Pre-compute inverse std for efficiency
            inv_std = ELT(1.0) / (bn.std_run[cidx] + IT)

            @turbo @. ch_z_norm = (ch_z - bn.mu_run[cidx]) * inv_std
            @turbo @. ch_z = ch_z_norm * bn.gam[cidx] + bn.bet[cidx]
        end
    end
    return
end


function batchnorm_grad!(layer::LinearLayer, current_batch_size)
    bn = layer.normparams
    cb = current_batch_size
    cb_rng = 1:cb
    mb = size(layer.eps_l, 2)

    if mb == cb
        eps_l = layer.eps_l
        z_norm = layer.z_norm
    else
        eps_l = view_minibatch(layer.eps_l, cb_rng)
        z_norm = view_minibatch(layer.z_norm, cb_rng)
    end

    inverse_mb_size = ELT(1.0) / ELT(cb)

    # replace one-liner with vectorized loop: bn.grad_bet .= sum(layer.eps_l, dims=2) .* inverse_mb_size  # ./ mb
    fill!(bn.grad_bet, ELT(0.0))
    @turbo for j in cb_rng   # axes(layer.eps_l, 2)
        for i in axes(layer.eps_l, 1)
            bn.grad_bet[j] += eps_l[i,j] * inverse_mb_size
        end
    end
    # replace one-liner with vectorized loop: bn.grad_gam .= sum(layer.eps_l .* layer.z_norm, dims=2)  .* inverse_mb_size  # ./ mb
    fill!(bn.grad_gam, ELT(0.0))
    @turbo for j in cb_rng  # axes(layer.eps_l, 2)
        for i in axes(layer.eps_l, 1)
            bn.grad_gam[j] += eps_l[i,j] * z_norm[i,j] * inverse_mb_size
        end
    end

    eps_l .= bn.gam .* eps_l  # often called dELTa_z_norm at this stage
    # often called dELTa_z, dx, dout, or dy
    eps_l .= (inverse_mb_size .* (ELT(1.0) ./ (bn.stddev .+ IT)) .*
                    (cb .* eps_l .- sum(eps_l, dims=2) .-
                    z_norm .* sum(eps_l .* z_norm, dims=2)))   # TODO replace sum with non-allocating approach
end


function batchnorm_grad!(layer::ConvLayer, layer_above, current_batch_size)
    bn = layer.normparams
    cb = current_batch_size
    cb_rng = 1:cb
    veps_l_above = view_minibatch(layer_above.eps_l, cb_rng)
    vzn = view_minibatch(layer.z_norm, cb_rng)
    vpa_eps_l = view_minibatch(layer.pad_above_eps, cb_rng)
    (_, _, c, _) = size(layer.pad_above_eps)
    inverse_mb_size = ELT(1.0) / ELT(cb)

    # Compute gradients for beta and gamma
    bn.grad_bet .= reshape(sum(veps_l_above, dims=(1, 2, 4)),c)  .* inverse_mb_size   # ./ mb
    bn.grad_gam .= reshape(sum(veps_l_above .* vzn, dims=(1, 2, 4)), c) .* inverse_mb_size  # ./ mb

    @inbounds @fastmath for (cidx, ch_z, ch_z_norm) in zip(1:c, eachslice(vpa_eps_l, dims=3), eachslice(vzn,dims=3))
        # Step 1: Scale by gamma (dELTa_z_norm)
        ch_z .= bn.gam[cidx] .* ch_z

        # Step 2: Compute statistics needed for gradient
        # - sum of dELTa_z_norm
        # - sum of dELTa_z_norm * z_norm
        ch_sum = ELT(0.0)
        ch_prod_sum = ELT(0.0)
        @turbo for i in eachindex(ch_z)
            ch_sum += ch_z[i]
            ch_prod_sum += ch_z[i] * ch_z_norm[i]
        end

        # Step 3: Compute final gradient (dELTa_z)
        # Formula: (1/mb) * (1/stddev) * (mb*dELTa_z_norm - sum(dELTa_z_norm) - z_norm*sum(dELTa_z_norm*z_norm))
        scale = ELT(1.0) / cb / (bn.stddev[cidx] + IT)
        @turbo for i in eachindex(ch_z)
            ch_z[i] = scale * (cb * ch_z[i] - ch_sum - ch_z_norm[i] * ch_prod_sum)
        end
    end
end

# =====================
# activation functions
# =====================

function relu!(layer, current_batch_size)
    cb = current_batch_size
    mb = size(layer.z, ndims(layer.z))
    cb_rng = 1:cb
    if mb == cb
        z = layer.z
        a = layer.a
    else
        z = view_minibatch(layer.z, cb_rng)
        a = view_minibatch(layer.a, cb_rng)
    end

    @turbo for i in eachindex(z)
        a[i] = ifelse(z[i] >= ELT(0.0), z[i], ELT(0.0)) # no allocations
    end
end

function leaky_relu!(layer, current_batch_size)
    cb = current_batch_size
    mb = size(layer.z, ndims(layer.z))
    cb_rng = 1:cb
    if mb == cb
        z = layer.z
        a = layer.a
    else
        z = view_minibatch(layer.z,cb_rng)
        a = view_minibatch(layer.a, cb_rng)
    end
    @turbo for i in eachindex(z)
        a[i] = ifelse(z[i] >= ELT(0.0), z[i], layer.adj * z[i]) # no allocations
    end
end


# use for activation of conv or linear, when activation is requested as :none
@inline noop(args...) = nothing


function relu_grad!(layer, current_batch_size)   # I suppose this is really leaky_relu...
    cb = current_batch_size
    mb = size(layer.z, ndims(layer.z))
    cb_rng = 1:cb
    if mb == cb
        z = layer.z
        grad_a = layer.grad_a
    else
        z = view_minibatch(layer.z, cb_rng)
        grad_a = view_minibatch(layer.grad_a, cb_rng)
    end
    @turbo for i = eachindex(z)  # when passed any array, this will update in place
        grad_a[i] = ifelse(z[i] > ELT(0.0), ELT(1.0), ELT(0.0))  # prevent vanishing gradients by not using 0.0f0
    end
end

function leaky_relu_grad!(layer, current_batch_size)   # I suppose this is really leaky_relu...
    cb = current_batch_size
    mb = size(layer.z, ndims(layer.z))
    cb_rng = 1:cb
    if mb == cb
        z = layer.z
        grad_a = layer.grad_a
    else
        z = view_minibatch(layer.z, cb_rng)
        grad_a = view_minibatch(layer.grad_a, cb_rng)
    end
    @turbo for i = eachindex(z)  # when passed any array, this will update in place
        grad_a[i] = ifelse(z[i] > ELT(0.0), ELT(1.0), layer.adj)  # prevent vanishing gradients by not using 0.0f0
    end
end


# =====================
# classifier and loss functions
# =====================

function dloss_dz!(layer, target, current_batch_size::Int) # TODO we assume this is always dense linear
    cb = current_batch_size
    mb = size(layer.z, ndims(layer.z))
    cb_rng = 1:cb

    # @show size(layer.eps_l)
    # @show size(layer.a)
    # @show size(target)

    if mb == cb
        layer.eps_l .= layer.a .- target    # no allocations                                                
    else
        @turbo @views layer.eps_l[:, cb_rng] .= layer.a[:, cb_rng] .- target  # [current_batch_size]
    end
end

# tested to have no allocations
function softmax!(layer, current_batch_size::Int)
    cb = current_batch_size
    for b in 1:cb   # axes(layer.z, 2)
        # Find maximum in this column
        max_val = typemin(ELT)
        @turbo for i in axes(layer.z, 1)
            max_val = max(max_val, layer.z[i, b])
        end

        # Compute exp and sum in one pass
        sum_exp = ELT(0.0)
        @turbo for i in axes(layer.z, 1)
            layer.a[i, b] = exp(layer.z[i, b] - max_val)
            sum_exp += layer.a[i, b]
        end

        # Normalize
        sum_exp = sum_exp + IT  # Add epsilon for numerical stability
        @turbo for i in axes(layer.z, 1)
            layer.a[i, b] /= sum_exp
        end
    end
    return
end

# TODO do we need a grad version?  of course, we do...
function logistic!(layer, current_batch_size)
    cb = current_batch_size
    cb_rng = 1:cb
    vz = view_minibatch(layer.z,cb_rng)
    va = view_minibatch(layer.a, cb_rng)

    @turbo va .= ELT(1.0) ./ (ELT(1.0) .+ exp.(.-vz))
end

# TODO  need to verify logic for this
function logistic_grad!(layer, current_batch_size)
    cb = current_batch_size
    cb_rng = 1:cb
    vz = view_minibatch(layer.z,cb_rng)
    vgrada = view_minibatch(layer.grad_a, cb_rng)
    #
    @turbo vgrada .= nothing
end


function sigmoid!(a::AbstractArray{Float64}, z::AbstractArray{Float64})
    @turbo @.a = 1.0 / (1.0 + exp(-z))  
end

function sigmoid_gradient!(grad::AbstractArray{Float64}, z::AbstractArray{Float64})
    sigmoid!(grad, z)
    @turbo @. grad = grad * (1.0 - grad)
end



function regression!(layer, current_batch_size)
    @turbo layer.a .= layer.z
end


# =============================
# optimization functions
# =============================


Base.@kwdef struct AdamParam <: OptParam
    b1::ELT
    b2::ELT
    decay::ELT  # for AdamW, often called lambda
end

struct NoOpt <: OptParam
    @inline NoOpt() = new()   # help compiler elide any call to empty constructor
end   # a noop struct when not doing Batch Normalization


# calculates the momentum 'm' and the root mean square 'v' term for adam and adamw
@inline function pre_adam!(layer, ad, t)
    # ad = layer.optparam
    # bn = layer.normparams

    adam_helper!(layer.grad_m_weight, layer.grad_v_weight, layer.grad_weight, ad, t)
    layer.dobias && (adam_helper!(layer.grad_m_bias, layer.grad_v_bias, layer.grad_bias, ad, t))
end

# TODO we should test for this earlier and not have to test again within the function
@inline function pre_adam_batchnorm!(bn, ad, t)
    adam_helper!(bn.grad_m_gam, bn.grad_v_gam, bn.grad_gam, ad, t)
    adam_helper!(bn.grad_m_bet, bn.grad_v_bet, bn.grad_bet, ad, t)
end

@inline function adam_helper!(grad_m_lrparam, grad_v_lrparam, grad_lrparam, ad, t)
    b1_term = ELT(1.0) - (ad.b1)^t
    b2_term = ELT(1.0) - (ad.b2)^t

    # Use @turbo for better performance
    @turbo for i in eachindex(grad_m_lrparam)
        grad_m_lrparam[i] = ad.b1 * grad_m_lrparam[i] + b1_term * grad_lrparam[i]
        grad_v_lrparam[i] = ad.b2 * grad_v_lrparam[i] + b2_term * grad_lrparam[i]^2
    end
end
