
# function reg_L1_update!(layer::Layer, hp, counter)
#     @inbounds for i in eachindex(layer.weight)
#         layer.weight[i] -= hp.lr * (layer.grad_weight[i] + hp.regparm * sign(layer.weight[i]))
#     end

#     # Separate loop for bias with no regularization
#     layer.dobias && @inbounds for i in eachindex(layer.bias)
#         layer.bias[i] -= hp.lr * layer.grad_bias[i]
#     end
# end

# TODO probably get rid of this and combine in one weight update function that can do everything
# function reg_L2_update!(layer::Layer, hp, counter)
#     @inbounds for i in eachindex(layer.weight)
#         layer.weight[i] -= hp.lr * (layer.grad_weight[i] + hp.regparm * layer.weight[i])
#     end

#     # Separate loop for bias with no regularization
#     layer.dobias && @inbounds for i in eachindex(layer.bias)
#         layer.bias[i] -= hp.lr * layer.grad_bias[i]
#     end
# end



# function simple_update!(layer::Layer, hp)
#     # use explicit loop to eliminate allocation and allow optimization
#     @inbounds for i in eachindex(layer.weight)
#         layer.weight[i] -= hp.lr * layer.grad_weight[i]
#     end

#     # Separate loop for bias
#     layer.dobias && @inbounds for i in eachindex(layer.bias)
#         layer.bias[i] -= hp.lr * layer.grad_bias[i]
#     end
# end


# function adam_update!(layer::Layer, hp, t)
#     ad = layer.optparams  

#     pre_adam!(layer, ad, t)
#     b1_divisor = 1.0 - ad.b1^t
#     b2_divisor = 1.0 - ad.b2^t

#     l2_term = (hp.reg == :L2) ? hp.lr * hp.regparm * layer.weight[i] : 0.0

#     @inbounds for i in eachindex(layer.weight)
#         layer.weight[i] -= hp.lr * (layer.grad_m_weight[i] / b1_divisor) / (sqrt(layer.grad_v_weight[i] / b2_divisor) + 1e-12) + l2_term
#     end

#     layer.dobias && @inbounds for i in eachindex(layer.bias)
#         layer.bias[i] -= hp.lr * (layer.grad_m_bias[i] / b1_divisor) / (sqrt(layer.grad_v_bias / b2_divisor) + 1e-12) 
#     end

#     # TODO where do we put the adam update of the batchnorm params?
# end


# function adamw_update!(layer::Layer, hp, t)
#     ad = layer.optparams

#     pre_adam!(layer, hp, t)
#     b1_divisor = 1.0 - ad.b1^t
#     b2_divisor = 1.0 - ad.b2^t

#     @inbounds for i in eachindex(layer.weight)
#         layer.weight[i] = (layer.weight[i] - 
#                             hp.lr * ((layer.grad_m_weight[i] / b1_divisor) / (sqrt(layer.grad_v_weight[i] / b2_divisor) + 1e-12)) 
#                             - hp.lr * ad.decay * layer.weight[i])
#     end

#     layer.dobias && @inbounds for i in eachindex(layer.bias)
#         layer.bias[i] = (layer.bias[i] - 
#                             hp.lr * ((layer.grad_m_bias[i] / b1_divisor) / (sqrt(layer.grad_v_bias[i] / b2_divisor) + 1e-12)) 
#                             - hp.lr * ad.decay * layer.bias[i])
#     end
# end
