using LinearAlgebra

# Type for holding convolution layer parameters
struct ConvLayer
    weights::Array{Float64,4}  # (height, width, in_channels, out_channels)
    bias::Vector{Float64} # 
    stride::Int
    pad::Int
end

struct ConvInput
    x_col::Matrix{Float64}
    w_col::Matrix{Float64}
    bias::Vector{Float64}
    out_channels::Int
    batch_size::Int
    output_height::Int
    output_width::Int
end

# image size for convolutional layers
struct ImgSize  
    height::Int
    width::Int
    channels::Int
    batch_size::Int
end

struct PoolLayer
    size::Int
    stride::Int
    mode::Symbol  # must be :max or :mean
end

struct FCLayer
    weights::Matrix{Float64}  # (out_features, in_features)
    bias::Vector{Float64}
end


# TODO handle asymmetric padding
function pad_array(x::Array{Float64,4}, pad::Int)
    height, width, channels, batch_size = size(x)
    result = zeros(Float64, height + 2pad, width + 2pad, channels, batch_size)
    result[pad+1:end-pad, pad+1:end-pad, :, :] = x
    return result
end

reshapeinplace(v,dims) = Base.__reshape((v,IndexLinear()),dims) # unsafe, non-allocating reshape


# TODO handle rectangular images and kernels
function im2col!(layerconv::ConvInput, x::Array{Float64,4}, layer::ConvLayer)

    height, width, in_channels, batch_size = size(x)
    
    # Add padding if needed
    if layer.pad > 0
        x = pad_array(x, layer.pad)  # expands size of x
    end
    # @show size(x)
    
    # output dimensions of convolution output image in geometric interpretation
    out_height = layerconv.ouput_height
    out_width = layerconv.output_width


    # Initialize column matrix: 
    #     num rows = kernelx by kernely by number of input_channels, 
    #     num columns = batch_size by out_height by out_width
    # col = zeros(Float64, kernel_size * kernel_size * in_channels, 
    #                     batch_size * out_height * out_width)

    x_col = layerconv.x_col
    
    # Fill column matrix
    col_idx = 1  # in the original code, num cols = b * img_rows * img_cols 
    for b in 1:batch_size
        for i in 1:out_height
            for j in 1:out_width
                h_start = (i-1) * stride + 1
                h_end = h_start + kernel_size - 1
                w_start = (j-1) * stride + 1
                w_end = w_start + kernel_size - 1
                
                # Extract patch, which includes padding if any
                patch = view(x, h_start:h_end, w_start:w_end, :, b:b) # 1 item of batch, all channels
                pos = 1
                # loops faster than broadcasting, no allocations:  col[:, col_idx] = vec(patch)  
                for c in axes(x,3)
                    for w in 1:kernel_size
                        for h in 1:kernel_size
                            x_col[pos, col_idx] = patch[h,w,c,1] # index rows down a column first
                            pos += 1
                        end
                    end
                end
                col_idx += 1
            end
        end
    end
    
    # return col
end


# recover the image stack from col format
function col2im(col::Array{Float64,2}, x_shape::ImgSize, kernel_size::Int, 
                stride::Int, pad::Int)
    # height, width, channels, batch_size = x_shape  # ht is n_rows, wd is n_cols
    k = kernel_size
    s = stride

    # Initialize output array with padding
    x = zeros(Float64, x_shape.height, x_shape.width, x_shape.channels, x_shape.batch_size)
    
    out_height = div((x_shape.height + 2pad - k), s) + 1
    out_width = div((x_shape.width + 2pad - k),s) + 1
    k2c = k*k*x_shape.channels  # height of col for one window

    for b in 1:x_shape.batch_size
        # Calculate starting column in col for this batch
        batch_start = (b-1) * out_height * out_width
        
        for i in 1:x_shape.height
            for j in 1:x_shape.width
                for c in 1:x_shape.channels
                    
                    i_pad = i + pad - 1
                    j_pad = j + pad - 1

                    # Get window position
                    col_idx = batch_start + (i_pad-1)*out_width + j_pad

                    # Calculate position within window where real pixel appears
                    window_col = 1 + pad  # Skip first padded column
                    window_row = 1 + pad  # Skip first padded row
                    row_idx = (window_col-1)*k + window_row + (c-1)*k*k

                    x[i,j,c,b] = col[row_idx, col_idx]
                end
            end
        end
    end
    
    return x 
end


function conv_dims(layer::ConvLayer, x::Array{Float64, 4})
    height, width, in_channels, batch_size = size(x)
    input_img_size = ImgSize(height, width, in_channels, batch_size)
    kernel_size = size(layer.weights)[1]

    # conv output image dims
    out_height = div((height + 2layer.pad - kernel_size), layer.stride) + 1
    out_width = div((width + 2layer.pad - kernel_size), layer.stride) + 1

    out_channels = size(layer.weights)[4]
    output_img_size = ImgSize(out_height, out_width, out_channels, batch_size)
    return input_img_size, output_img_size  
end

# used to determine pooling output dims in advance to pre-allocate storage array
function pool_dims(layer::PoolLayer, x::Array{Float64,4})
    h, w, c, batch = size(x)
    out_h = div((h - layer.size), layer.stride) + 1
    out_w = div((w - layer.size), layer.stride) + 1
    return ImgSize(out_h, out_w, c, batch)
end

# to move im2col out of forwardconv function
function prep_forward!(layerconv::ConvInput, layer::ConvLayer, x::Array{Float64,4})
    input_img_size, output_img_size = conv_dims(layer, x)
    kernel_size = size(layer.weights)[1]
    out_channels = output_img_size.channels
        
    # Convert input to column format
    # x_col = im2col(x, kernel_size, layer.stride, layer.pad)
    im2col!(layerconv, x, layer)  # to calculate layerconv.x_col
    # println("size of x_col ", size(x_col))
    
    # Reshape weights for matrix multiplication
    layerconv.w_col .= reshapeinplace(layer.weights, (kernel_size * kernel_size * input_img_size.channels, out_channels))
    # @show size(w_col)   
    # @show typeof(w_col)
    layerconv.bias .= layer.bias
    # conv_inputs = ( x_col, w_col, bias, out_channels, input_img_size.batch_size, output_img_size.height, output_img_size.width)
    # conv_input = ConvInput( x_col, w_col, bias, out_channels, input_img_size.batch_size, output_img_size.height, output_img_size.width)

    # TODO should we return input or output img_size???
    # return (; conv_input, input_img_size)  # named tuples are typed; better than Python-esque untyped cache
end

# this method is passed the converted column format image, weights, and bias, etc.
function forward_conv(x_col::Array{Float64,2}, w_col::Array{Float64,2}, bias::Array{Float64,1},
            out_channels, batch_size, out_h, out_w)

    raw_out = w_col' * x_col  # convolution as matrix multiplication
    out = zeros(out_h, out_w, out_channels, batch_size)  # initialize img stack
    # rearrange raw_out to image stack
    @inbounds for c in 1:out_channels   # channels are rows with 1 or more batches across
        for b in 1:batch_size    # batches
            for h in 1:out_h     # row of image across
                for w in 1:out_w  # column values of each row of the image
                    #   for each channel row: batch of images, image plane row by columns
                    out[h,w,c,b] = raw_out[c, ((b-1) * out_h * out_w) + ((h-1) * out_h) + w]
                end
            end
        end
    end

    # add bias to all values of each channel
    @inbounds for b in axes(out,4)
                for c in axes(out,3)
                    for j in axes(out,2)
                        for i in axes(out,1)
                            out[i, j, c, b] += bias[c] 
                        end
                    end
                end
            end
    
    return out      
end

function forward_pool(x::Array{Float64,4}, layer::PoolLayer)
    h, w, c, batch = size(x)
    out_h = div((h - layer.size), layer.stride) + 1
    out_w = div((w - layer.size), layer.stride) + 1
    
    out = zeros(Float64, out_h, out_w, c, batch)
    locations = fill(CartesianIndex(0,0), out_h, out_w, c, batch)
    # @show size(locations)
    
    for b in 1:batch, c in 1:c
        for i in 1:out_h, j in 1:out_w
            h_start = (i-1) * layer.stride + 1
            w_start = (j-1) * layer.stride + 1
            window = view(x, h_start:h_start+layer.size-1,
                        w_start:w_start+layer.size-1, c, b)
            if layer.mode == :max
                out[i,j,c,b], idx = findmax(window)
                # @show out[i,j,c,b]
                # @show idx
                locations[i,j,c,b] = idx
            elseif layer.mode == :mean
                out[i,j,c,b] = mean(window)
            else
                error("Wrong mode in PoolLayer: " * string(layer.mode))
            end
        end
    end
    
    return out, locations
end

function forward_pool!(out::Array{Float64,4}, locs::Array{CartesianIndex{2},4}, x::Array{Float64,4}, layer::PoolLayer)
    h, w, c, batch = size(x)
    out_h = div((h - layer.size), layer.stride) + 1
    out_w = div((w - layer.size), layer.stride) + 1
        
    for b in 1:batch, c in 1:c
        for i in 1:out_h, j in 1:out_w
            h_start = (i-1) * layer.stride + 1
            w_start = (j-1) * layer.stride + 1
            window = view(x, h_start:h_start+layer.size-1,
                        w_start:w_start+layer.size-1, c, b)
            if layer.mode == :max
                out[i,j,c,b], idx = findmax(window)
                locs[i,j,c,b] = idx
            elseif layer.mode == :mean
                out[i,j,c,b] = mean(window)
            else
                error("Wrong mode in PoolLayer: " * string(layer.mode))
            end
        end
    end
end

function forward_fc(layer::FCLayer, x::Matrix{Float64})
    return layer.weights * x .+ layer.bias, x
end


# should we pass weights or layer struct?
# function forward(layer::ConvLayer, x::Array{Float64,4})
#     height, width, in_channels, batch_size = size(x)
#     # @show in_channels
#     kernel_size = size(layer.weights)[1]
#     out_channels = size(layer.weights)[4]
    
#     out_height = (height + 2layer.pad - kernel_size) ÷ layer.stride + 1
#     out_width = (width + 2layer.pad - kernel_size) ÷ layer.stride + 1
    
#     # Convert input to column format
#     x_col = im2col(x, kernel_size, layer.stride, layer.pad)
#     # @show size(x_col)
    
#     # Reshape weights for matrix multiplication
#     w_col = reshape(layer.weights, kernel_size * kernel_size * in_channels, out_channels)
#     # @show size(w_col)    

#     # Perform convolution as matrix multiplication
#     out = w_col' * x_col
#     # out = x_col' * w_col
#     # @show size(out)
    
#     # Reshape output
#     out = reshape(out, out_channels, batch_size, out_height, out_width) 
#     out = permutedims(out, (4,3,1,2))  # transpose the 2d image # Move batch_size to last dimension
    
#     # Add bias
#     for i in axes(out,3)
#         out[:,:,i,:] .+= layer.bias[i]
#     end

#     # out is the image_stack post convolution, x is the input image stack, x_col is unrolled, w is unrolled weights
#     return out     #, (x, x_col, w_col)  
# end

# TODO eliminate duplicate inputs for convolution and image size
function backward_conv(layer::ConvLayer, dL_dout::Array{Float64,4}, input_img_size::ImgSize, conv_input::ConvInput)

    x_col = conv_input.x_col
    w_col = conv_input.w_col
    height = input_img_size.height
    width = input_img_size.width
    in_channels = input_img_size.channels
    batch_size = input_img_size.batch_size
    kernel_size = size(layer.weights)[1]
    out_channels = size(layer.weights)[4]
    
    # Reshape gradient for matrix multiplication
    dL_dout_reshaped = reshape(dL_dout, out_channels, :)

    # Compute gradients
    dL_dw_col = dL_dout_reshaped * x_col'
    dL_dw = reshape(dL_dw_col, kernel_size, kernel_size, in_channels, out_channels)
    
    dL_dx_col = w_col * dL_dout_reshaped
    dL_dx = col2im(dL_dx_col, input_img_size, kernel_size, layer.stride, layer.pad)  
    
    # Compute bias gradient
    # @show size(dL_dout)
    dL_db = vec(sum(dL_dout; dims=(1,2,4)))
    
    return dL_dx, dL_dw, dL_db
end

function backward_pool(layer::PoolLayer, dL_dout::Array{Float64,4}, 
                        x_shape::Tuple, locations::Array{CartesianIndex{2},4})
    h, w, c, batch = x_shape  # the input in the forward pass--bigger!
    dL_dx = zeros(Float64, h, w, c, batch)  # the output going down from backprop
    
    out_h, out_w = size(dL_dout)[1:2]
    
    if layer.mode == :max
        for b in 1:batch, c in 1:c
            for i in 1:out_h, j in 1:out_w
                h_start = (i-1) * layer.stride + 1
                w_start = (j-1) * layer.stride + 1
                idx = locations[i,j,c,b]
                h_offset, w_offset = idx[1],idx[2]
                dL_dx[h_start+h_offset-1, w_start+w_offset-1, c, b] = dL_dout[i,j,c,b]
            end
        end
    elseif layer.mode == :mean
        for b in 1:batch, c in 1:c
            for i in 1:out_h, j in 1:out_w
                h_start = (i-1) * layer.stride + 1
                w_start = (j-1) * layer.stride + 1
                window = view(dL_dx, h_start:h_start+layer.size-1, w_start:w_start+layer.size-1, c, b)
                window .= dL_dout[i,j,c,b]  # put the mean in all the cells in the window
            end
        end
    else
        error("Wrong mode in PoolLayer: " * layer.mode)
    end
    
    return dL_dx
end

function backward_fc(layer::FCLayer, dL_dout::Matrix{Float64}, cache)
    x = cache
    dL_dw = dL_dout * x'
    dL_db = vec(sum(dL_dout, dims=2))
    dL_dx = layer.weights' * dL_dout
    return dL_dx, dL_dw, dL_db
end

# Transition functions  
function flatten_img(x::Array{Float64,4})
    # Reshape from (h, w, c, batch) to (h*w*c, batch)
    return reshape(x, prod(size(x)[1:3]), :)  #  275 nanosecs
end

function unflatten_gradient(grad::Matrix{Float64}, conv_shape::Tuple)
    # Reshape from (h*w*c, batch) to (h, w, c, batch)
    h, w, c, batch = conv_shape
    return permutedims(reshape(grad, h, w, c, batch), (1,2,3,4))
end

# THIS IS TEMPORARY UNTIL USING THE EXISTING FRAMEWORK
function he_initialize(wgts::AbstractArray)
        n_in = sum(size(wgts))
        scale = 2.0 / n_in
        randn(size(wgts)) .* sqrt(scale)
end

function he_initialize(wgt_dims::Tuple)
        # @show wgt_dims
        n_in = sum(wgt_dims)
        scale = 2.0 / n_in
        randn(wgt_dims) .* sqrt(scale)
end

function he_initialize(w...)
    he_initialize(w)
end


# Test function
function test_conv()
    # Create small test case
    batch_size, height, width = 2, 6, 6  # Smaller size for testing
    in_channels, out_channels = 2, 3
    kernel_size = 3
    stride = 1
    pad = 1
    
    # Initialize layer
    weights = randn(Float64, kernel_size, kernel_size, in_channels, out_channels)
    bias = randn(Float64, out_channels)
    # @show size(bias)
    layer = ConvLayer(weights, bias, stride, pad)
    
    # Create input
    x = randn(Float64,  height, width, in_channels, batch_size)
    
    # Forward pass
    println("Running forward pass...")
    conv_inputs, input_img_size = prep_forward(layer,x)
    out = forward_conv(conv_inputs...)
    println("Forward pass output shape: ", size(out))
    
    # Backward pass
    println("\nRunning backward pass...")
    dL_dout = randn(Float64, size(out))
    dL_dx, dL_dw, dL_db = backward_conv(layer, dL_dout, input_img_size, conv_inputs)
    
    println("\nGradient shapes:")
    println("dL_dx: ", size(dL_dx))
    println("dL_dw: ", size(dL_dw))
    println("dL_db: ", size(dL_db))
    
    # Verify shapes
    @assert size(dL_dx) == size(x) "Input gradient shape mismatch"
    @assert size(dL_dw) == size(weights) "Weight gradient shape mismatch"
    @assert size(dL_db) == size(bias) "Bias gradient shape mismatch"
    
    println("\nAll shape checks passed!")
    return true
end

#=
The output from w_col' * x_col will have dimensions (out_channels × (batch_size * out_height * out_width)). So for each output channel i, we need to add the corresponding bias[i] to all columns in row i of this matrix.
This makes adding the bias pretty simple - we can just add the bias vector to each batch/spatial position in the output by broadcasting. In Julia this would be something like adding the bias (reshaped to match the output dimensions) to each column.
But let me double check my matrix dimensions:

x_col is (kernel_size * kernel_size * in_channels) × (batch_size * out_height * out_width)
w_col is (kernel_size * kernel_size * in_channels) × out_channels
w_col' * x_col gives us (out_channels × (batch_size * out_height * out_width))

So yes, each row i contains all the outputs for output channel i, spread across all batches and spatial positions. The bias for channel i needs to be added to all entries in row i.

=#
