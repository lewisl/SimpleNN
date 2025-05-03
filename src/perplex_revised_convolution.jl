function test_conv_dimensions(layer, layer_above)
    f_h, f_w = size(layer.weight)[1:2]
    
    # Expected dimensions after backprop
    expected_H = size(layer_above.eps_l, 1) + (f_h - 1)
    expected_W = size(layer_above.eps_l, 2) + (f_w - 1)
    
    # Actual dimensions of layer.eps_l
    actual_H, actual_W = size(layer.eps_l)[1:2]
    
    println("Expected dimensions: $(expected_H)×$(expected_W)")
    println("Actual dimensions: $(actual_H)×$(actual_W)")
    println("Match: $(expected_H == actual_H && expected_W == actual_W)")
end

function check_conv_gradients(layer, layer_above, input_data, epsilon=1e-7)
    # Store original weights
    original_weights = copy(layer.weight)
    
    # Forward pass to get original loss
    # (Assuming you have a forward function and loss calculation)
    forward!(layer, input_data)
    forward!(layer_above, layer.a)
    original_loss = calculate_loss(layer_above.a, targets)
    
    # Backprop to get analytical gradients
    layer_backward!(layer, layer_above)
    analytical_grads = copy(layer.grad_weight)
    
    # Numerical gradients
    numerical_grads = similar(layer.weight)
    
    # Check a few random weights
    for idx in rand(1:length(layer.weight), 10)
        # Perturb weight positively
        layer.weight[idx] += epsilon
        forward!(layer, input_data)
        forward!(layer_above, layer.a)
        loss_plus = calculate_loss(layer_above.a, targets)
        
        # Perturb weight negatively
        layer.weight[idx] -= 2*epsilon
        forward!(layer, input_data)
        forward!(layer_above, layer.a)
        loss_minus = calculate_loss(layer_above.a, targets)
        
        # Restore weight
        layer.weight[idx] += epsilon
        
        # Numerical gradient
        numerical_grads[idx] = (loss_plus - loss_minus) / (2 * epsilon)
        
        println("Weight $idx: Analytical: $(analytical_grads[idx]), Numerical: $(numerical_grads[idx])")
    end
    
    # Restore original weights
    layer.weight .= original_weights
end

function test_loop_bounds(layer, layer_above)
    f_h, f_w = size(layer.weight)[1:2]
    H_out, W_out = size(layer_above.eps_l)[1:2]
    
    # Initialize with recognizable pattern
    layer_above.eps_l .= 1.0
    layer.weight .= 1.0
    layer.eps_l .= 0.0
    
    # Run your backprop function
    layer_backward!(layer, layer_above)
    
    # Check if all expected elements received updates
    zeros_count = count(x -> x == 0.0, layer.eps_l)
    total_elements = length(layer.eps_l)
    
    println("Elements that received updates: $(total_elements - zeros_count) out of $total_elements")
    
    # Check the pattern of updates
    println("First few rows of first channel:")
    display(layer.eps_l[1:10, 1:10, 1, 1])
    
    # Check if boundary elements were updated as expected
    boundary_updated = any(layer.eps_l[end-f_h+2:end, :, :, :] .!= 0) && 
                      any(layer.eps_l[:, end-f_w+2:end, :, :] .!= 0)
    println("Boundary elements updated: $boundary_updated")
end



function layer_backward!(layer::ConvLayer, layer_above)
    (f_h, f_w, in_channels, out_channels) = size(layer.weight)
    (H_below, W_below, _, batch_size) = size(layer.a_below)
    (H_out, W_out, _, _) = size(layer_above.eps_l)
    
    # Initialize gradients
    layer.grad_weight .= 0.0
    layer.eps_l .= 0.0
    
    # Calculate grad_weight (straightforward)
    for b = 1:batch_size
        for oc = 1:out_channels
            for i = 1:H_out
                for j = 1:W_out
                    for ic = 1:in_channels
                        for fi = 1:f_h
                            for fj = 1:f_w
                                layer.grad_weight[fi,fj,ic,oc] += layer.a_below[i+fi-1,j+fj-1,ic,b] * layer_above.eps_l[i,j,oc,b]
                            end
                        end
                    end
                end
            end
        end
    end
    
    # Calculate layer.eps_l using full convolution
    # For each position in the output (eps_l), we need to find all contributions from the error
    for b = 1:batch_size
        for ic = 1:in_channels
            for i = 1:H_below
                for j = 1:W_below
                    for oc = 1:out_channels
                        # For each position in eps_l, find all overlapping filter positions
                        # from the error map (layer_above.eps_l)
                        for fi = 1:f_h
                            for fj = 1:f_w
                                # Calculate the position in layer_above.eps_l that would use this position
                                # during the forward pass with this filter position
                                i_out = i - (fi - 1)
                                j_out = j - (fj - 1)
                                
                                # Only add contribution if within bounds of layer_above.eps_l
                                if 1 <= i_out <= H_out && 1 <= j_out <= W_out
                                    layer.eps_l[i,j,ic,b] += layer.weight[f_h-fi+1,f_w-fj+1,ic,oc] * layer_above.eps_l[i_out,j_out,oc,b]
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    
    # Apply activation derivative
    relu_grad!(layer)
    layer.eps_l .*= layer.grad_a
    
    # Calculate grad_bias
    layer.grad_bias .= 0.0
    for oc = 1:out_channels
        layer.grad_bias[oc] = sum(layer.eps_l[:,:,oc,:])
    end
    
    return  # nothing
end
