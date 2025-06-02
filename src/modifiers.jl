
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


function batchnorm!(layer::LinearLayer)
    bn = layer.normparams

    if bn.istraining[]        # access value of Ref, like a 1 element array
        mean!(bn.mu, layer.z)
        bn.stddev .= std(layer.z, dims=2)

        @turbo @. layer.z_norm = (layer.z - bn.mu) / (bn.stddev + IT) # normalized: often xhat or zhat
        @turbo @. layer.z = layer.z_norm * bn.gam + bn.bet  # shift & scale: often called y
        @. bn.mu_run = ifelse(bn.mu_run[1] == ELT(0.0), bn.mu, ELT(0.95) * bn.mu_run + ELT(0.05) * bn.mu)
        @. bn.std_run = ifelse(bn.std_run[1] == ELT(0.0), bn.stddev, ELT(0.95) * bn.std_run + ELT(0.05) * bn.stddev)
    else  # prediction: use running mean and stddev
        @turbo @. layer.z_norm = (layer.z - bn.mu_run) / (bn.std_run + IT) # normalized: aka xhat or zhat
        @turbo @. layer.z = layer.z_norm * bn.gam + bn.bet  # shift & scale: often called y
    end
    return
end

function batchnorm!(layer::ConvLayer)
    bn = layer.normparams
    c = size(layer.z, 3)
    if bn.istraining[]    # access value of Ref, like a 1 element array
        @inbounds @fastmath for (cidx, ch_z, ch_z_norm) in zip(1:c, eachslice(layer.z, dims=3), eachslice(layer.z_norm, dims=3))
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
        @inbounds @fastmath for (cidx, ch_z, ch_z_norm) in zip(1:c, eachslice(layer.z, dims=3), eachslice(layer.z_norm, dims=3))
            # Pre-compute inverse std for efficiency
            inv_std = ELT(1.0) / (bn.std_run[cidx] + IT)

            @turbo @. ch_z_norm = (ch_z - bn.mu_run[cidx]) * inv_std
            @turbo @. ch_z = ch_z_norm * bn.gam[cidx] + bn.bet[cidx]
        end
    end
    return
end


function batchnorm_grad!(layer::LinearLayer)
    bn = layer.normparams
    mb = size(layer.eps_l, 2)
    inverse_mb_size = ELT(1.0) / ELT(mb)

    # replace one-liner with vectorized loop: bn.grad_bet .= sum(layer.eps_l, dims=2) .* inverse_mb_size  # ./ mb
    fill!(bn.grad_bet, ELT(0.0))
    @turbo for j in axes(layer.eps_l, 2)
        for i in axes(layer.eps_l, 1)
            bn.grad_bet[j] += layer.eps_l[i,j] * inverse_mb_size
        end
    end
    # replace one-liner with vectorized loop: bn.grad_gam .= sum(layer.eps_l .* layer.z_norm, dims=2)  .* inverse_mb_size  # ./ mb
    fill!(bn.grad_gam, ELT(0.0))
    @turbo for j in axes(layer.eps_l, 2)
        for i in axes(layer.eps_l, 1)
            bn.grad_gam[j] += layer.eps_l[i,j] * layer.z_norm[i,j] * inverse_mb_size
        end
    end

    layer.eps_l .= bn.gam .* layer.eps_l  # often called dELTa_z_norm at this stage
    # often called dELTa_z, dx, dout, or dy
    layer.eps_l .= (inverse_mb_size .* (ELT(1.0) ./ (bn.stddev .+ IT)) .*
                    (mb .* layer.eps_l .- sum(layer.eps_l, dims=2) .-
                    layer.z_norm .* sum(layer.eps_l .* layer.z_norm, dims=2)))   # replace with non-allocating approach
end


function batchnorm_grad!(layer::ConvLayer)
    bn = layer.normparams
    (_, _, c, mb) = size(layer.pad_above_eps)
    inverse_mb_size = ELT(1.0) / ELT(mb)

    # @show size(layer.z_norm)
    # @show size(layer.pad_above_eps)


    # Compute gradients for beta and gamma
    bn.grad_bet .= reshape(sum(layer.pad_above_eps, dims=(1, 2, 4)),c)  .* inverse_mb_size   # ./ mb
    bn.grad_gam .= reshape(sum(layer.pad_above_eps .* layer.z_norm, dims=(1, 2, 4)), c) .* inverse_mb_size  # ./ mb

    @inbounds @fastmath for (cidx, ch_z, ch_z_norm) in zip(1:c, eachslice(layer.pad_above_eps, dims=3), eachslice(layer.z_norm,dims=3))
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
        scale = ELT(1.0) / mb / (bn.stddev[cidx] + IT)
        @turbo for i in eachindex(ch_z)
            ch_z[i] = scale * (mb * ch_z[i] - ch_sum - ch_z_norm[i] * ch_prod_sum)
        end
    end
end

# =====================
# activation functions
# =====================

function relu!(layer)
    # @inbounds @fastmath begin
    @turbo for i in eachindex(layer.z)
        layer.a[i] = ifelse(layer.z[i] >= ELT(0.0), layer.z[i], ELT(0.0)) # no allocations
    end
    # end
end

function leaky_relu!(layer)
    # @inbounds @fastmath begin
        @turbo for i in eachindex(layer.z)
            layer.a[i] = ifelse(layer.z[i] >= ELT(0.0), layer.z[i], layer.adj * layer.z[i]) # no allocations
        end
    # end
end


# use for activation of conv or linear, when activation is requested as :none
@inline noop(args...) = nothing


function relu_grad!(layer)   # I suppose this is really leaky_relu...
    @turbo for i = eachindex(layer.z)  # when passed any array, this will update in place
        layer.grad_a[i] = ifelse(layer.z[i] > ELT(0.0), ELT(1.0), ELT(0.0))  # prevent vanishing gradients by not using 0.0f0
    end
end

function leaky_relu_grad!(layer)   # I suppose this is really leaky_relu...
    @turbo for i = eachindex(layer.z)  # when passed any array, this will update in place
        layer.grad_a[i] = ifelse(layer.z[i] > ELT(0.0), ELT(1.0), layer.adj)  # prevent vanishing gradients by not using 0.0f0
    end
end


# =====================
# classifier and loss functions
# =====================

function dloss_dz!(layer, target)
    layer.eps_l .= layer.a .- target
end

# tested to have no allocations
function softmax!(layer)
    for c in axes(layer.z, 2)
        # Find maximum in this column
        max_val = typemin(ELT)
        @turbo for i in axes(layer.z, 1)
            max_val = max(max_val, layer.z[i, c])
        end

        # Compute exp and sum in one pass
        sum_exp = ELT(0.0)
        @turbo for i in axes(layer.z, 1)
            layer.a[i, c] = exp(layer.z[i, c] - max_val)
            sum_exp += layer.a[i, c]
        end

        # Normalize
        sum_exp = sum_exp + IT  # Add epsilon for numerical stability
        @turbo for i in axes(layer.z, 1)
            layer.a[i, c] /= sum_exp
        end
    end
    return
end

function logistic!(layer)
    @fastmath layer.a .= ELT(1.0) ./ (ELT(1.0) .+ exp.(.-layer.z))
end

function regression!(layer)
    layer.a[:] = layer.z[:]
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
