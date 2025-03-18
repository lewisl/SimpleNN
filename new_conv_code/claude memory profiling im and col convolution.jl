using LinearAlgebra, Profile

# Original image convolution implementation
function convolve_multi(img::Array{Float32,4}, fil::Array{Float32,4}; stri=1)
    batch_size, in_channels, imgx, imgy = size(img)
    kernel_size, _, _, out_channels = size(fil)
    
    # Calculate output dimensions
    x_out = (imgx - kernel_size) ÷ stri + 1
    y_out = (imgy - kernel_size) ÷ stri + 1
    
    ret = zeros(Float32, batch_size, out_channels, x_out, y_out)
    
    for b = 1:batch_size
        for oc = 1:out_channels
            for j = 1:y_out
                for i = 1:x_out
                    element = 0.0f0
                    for ic = 1:in_channels
                        for kj = 1:kernel_size
                            for ki = 1:kernel_size
                                element += img[b,ic,stri*(i-1)+ki,stri*(j-1)+kj] * 
                                        fil[ki,kj,ic,oc]
                            end
                        end
                    end
                    ret[b,oc,i,j] = element
                end
            end
        end
    end
    return ret
end

# im2col version--a better version chosen
function im2col(x::Array{Float64,4}, kernel_size::Int, stride::Int)
    batch_size, in_channels, height, width = size(x)
    
    # Calculate output dimensions
    out_height = (height - kernel_size) ÷ stride + 1
    out_width = (width - kernel_size) ÷ stride + 1
    
    # Initialize column matrix
    col = zeros(Float64, kernel_size * kernel_size * in_channels, 
                        batch_size * out_height * out_width)
    
    # Fill column matrix
    col_idx = 1 # funky way to do parallel indices. max val = batch_size * out_height * out_width
    for b in 1:batch_size
        for i in 1:out_height
            for j in 1:out_width
                h_start = (i-1) * stride + 1
                h_end = h_start + kernel_size - 1
                w_start = (j-1) * stride + 1
                w_end = w_start + kernel_size - 1
                
                # Extract and flatten patch
                for ic in 1:in_channels
                    for ki in 1:kernel_size
                        for kj in 1:kernel_size
                            row_idx = ((ic-1) * kernel_size + ki-1) * kernel_size + kj
                            col[row_idx, col_idx] = x[b,ic,h_start+ki-1,w_start+kj-1]
                        end
                    end
                end
                col_idx += 1
            end
        end
    end
    return col
end

function conv_col(x::Array{Float64,4}, weights::Array{Float64,4}, stride::Int)
    kernel_size = size(weights, 1)
    out_channels = size(weights, 4)
    
    # Convert input to column format
    x_col = im2col(x, kernel_size, stride)
    
    # Reshape weights
    w_col = reshape(weights, :, out_channels)
    
    # Perform convolution as matrix multiplication
    out = w_col' * x_col
    
    # Reshape output to image format: NO it's already in image format
    # batch_size, in_channels, height, width = size(x)
    # out_height = (height - kernel_size) ÷ stride + 1
    # out_width = (width - kernel_size) ÷ stride + 1
    
    return reshape(out, out_channels, batch_size, out_height, out_width)
end

function profile_convolution_memory()
    # Setup test data
    batch_size, height, width = 32, 28, 28
    in_channels, out_channels = 3, 16
    kernel_size = 3
    stride = 1
    
    # Initialize data
    x = randn(Float32, batch_size, in_channels, height, width)
    weights = randn(Float32, kernel_size, kernel_size, in_channels, out_channels)
    
    # Profile im2col version
    println("\nProfiling col version:")
    col_alloc = @allocated begin
        for i in 1:100
            out = conv_col(x, weights, stride)
        end
    end
    
    # Profile direct convolution version
    println("\nProfiling im version:")
    im_alloc = @allocated begin
        for i in 1:100
            out = convolve_multi(x, weights, stri=stride)
        end
    end
    
    # Print results
    println("\nCol format total bytes: ", col_alloc)
    println("Im format total bytes: ", im_alloc)
    
    # Memory for one pass
    println("\nMemory for single forward pass:")
    println("Col format single pass: ", @allocated conv_col(x, weights, stride))
    println("Im format single pass: ", @allocated convolve_multi(x, weights, stri=stride))
end

# Run the profiling
profile_convolution_memory()