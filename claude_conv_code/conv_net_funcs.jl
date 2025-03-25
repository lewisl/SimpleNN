# TODO 
#   cut the piece in loops rather than using slicing notation
#   do examples loop across layers not per layer
#   implement "no check" versions for convolve and other image functions
#   is it important to allow asymmetric stride?   not yet....
#   support custom padding, not just :same
#   create new_img_size function for pooling output



"""
Convolve a one or multi-channel image with a filter with one or more output channels.
This is a 20x speedup over array broadcasting. Requires pre-allocated padded image and output image.

padding must be :same, :none, or :full.   :none is often called :valid...
padsym must be :sym or :asym. fil_idx must be an anonymous function that sets the 
filter index for the convolution loop.
"""
@views function convolve_multi!(img_out, img_in, img_padded, fil, bias; allpads, stride=1, for_grad=false)  
    if ndims(img_out) == 4
        imgx, imgy, imgc, n = size(img_out)
    elseif ndims(img_out) == 3
        imgx, imgy, imgc = size(img_out)
        n = 1
    elseif ndims(img_out)== 2
        imgx, imgy = size(img_out)
        imgc = 1
        n = 1
    else
        error("Image stack must have 2 or 3 dimensions.")
    end

    if ndims(img_padded) == 4
        x_pad, y_pad, imgc, n = size(img_padded)
    elseif ndims(img_padded) == 3
        x_pad, y_pad, imgc = size(img_padded)
        n = 1
    elseif ndims(img_padded)== 2
        x_pad, y_pad = size(img_padded)
        imgc = 1
        n = 1
    else
        error("Image stack must have 2 or 3 dimensions.")
    end

    @show allpads[1]+1:x_pad-allpads[2], allpads[3]+1:y_pad-allpads[4]
    img_padded[allpads[1]+1:x_pad-allpads[2], allpads[3]+1:y_pad-allpads[4],:,:] = img_in

    # set sizes of image and filter
    if ndims(fil) == 2
        filx, fily = size(fil)
        filc = 1
        fil = reshape(fil,filx, fily, filc)
    elseif ndims(fil) == 3  # muliple filters apply to all input image channels and create multiple output feature maps
        filx, fily, filc = size(fil)
    else
        error("wrong number of dimensions for filter: $(ndims(fil))")
    end

    for ci in 1:n
        for z in 1:filc  # new channels based on filters (e.g.--weights)
            for (j_out, j_img) in zip(1:imgy, 1:stride:y_pad)  # column major access
                for (i_out, i_img) in zip(1:imgx, 1:stride:x_pad) # 1st steps through ret; 2nd steps through image subset
                    element = 0.0
                    for ic = 1:imgc, fi = 1:filx, fj = 1:fily   # input image channels  # scalar multiply faster than slice & broadcast
                        img_i = i_img + fi - 1
                        img_j = j_img + fj - 1
                        element += img_padded[img_i, img_j, ic, ci] * fil[fi, fj, z]
                    end
                    img_out[i_out,j_out, z, n] = element + bias[z]
                end
            end
        end
    end
end

function padding_calc(imgsize, filsize, stride)


        @show typeof(stride), stride


    p = (filsize - imgsize - stride + imgsize*stride)
    p1 = div(p, 2) # p1 = p2 if imgsize is odd
    p2 = p - p1
    return p1, p2
end

"""
Convolve a one or multi-channel image with a filter with one or more output channels.
This is a 20x speedup over array broadcasting.

padding must be :same, :none, or :full.   :none is often called :valid...
padsym must be :sym or :asym. fil_idx must be an anonymous function that sets the 
filter index for the convolution loop.
"""
function convolve_multi(img, fil, bias; stride=1, padding=:same, padsym=:sym, fil_idx::Function = (maxidx, idx) -> idx)  
    if ndims(img) == 3
        imgx, imgy, imgc = size(img)
    elseif ndims(img)== 2
        imgx, imgy = size(img)
        imgc = 1
    else
        error("Image slice must have 2 or 3 dimensions.")
    end

    # set sizes of image and filter
    if ndims(fil) == 2
        filx, fily = size(fil)
        filc = filp = 1
        fil = reshape(fil,filx, fily, filc, filp)
    elseif ndims(fil) == 3  # one filter to match multiple input channels
        filx, fily, filc = size(fil)
        filp = 1
        fil = reshape(fil,filx, fily, filc, filp)       
    elseif ndims(fil) == 4  # multiple filters
        filx, fily, filc, filp = size(fil)  # filc = filter channels must equal image channels; filp = filter planes--number of output channels
    else
        error("wrong number of dimensions for filter: $(ndims(fil))")
    end


    println("stride = ", stride)


    # set padding for :same, :none, or :full
    if padding == :same
        pad_r1, pad_r2 = padding_calc(imgx, filx, stride)
        pad_c1, pad_c2 = padding_calc(imgy, fily, stride)
        if padsym == :sym
            img_padded = dopad(img, pad_r1)
        else
            img_padded = dopad(img, pad_r1, padc=pad_c2)
        end
    else # treat like :none for now...  TODO
        # dimensions of the single plane convolution result
        pad_r1, pad_r2 = 0,0
        pad_c1, pad_c2 = 0,0
        img_padded = img
    end

    # calculate output dimensions
    x_out = floor(Int, (imgx - filx + pad_r1 + pad_r2) / stride) + 1
    y_out = floor(Int, (imgy - fily + pad_c1 + pad_c2) / stride) + 1

    ret = zeros(x_out, y_out, filp)
    for z in 1:filp  # new channels
        for (j_out, j_img) in zip(1:y_out, 1:stride:imgy)  # column major access
            for (i_out, i_img) in zip(1:x_out, 1:stride:imgx) # 1st steps through ret; 2nd steps through image subset
                element = 0.0
                for ic = 1:imgc, fj = 1:fily, fi = 1:filx  # input image channels  # scalar multiply faster than slice & broadcast
                    img_i = i_img + fi - 1
                    img_j = j_img + fj - 1
                    element += img_padded[img_i, img_j, ic] * fil[fil_idx(filx,fi), fil_idx(fily,fj), ic, z]

                end
                ret[i_out,j_out, z] = element += bias[z]
            end
        end
    end
    return ret
end


function convolve_multi_grad(img, fil; stride=1, padding=:same, padsym=:sym)

    # Initialize gradients
    dl_dimg = zeros(size(img))
    dl_dfil = zeros(size(fil))

    filtup = size(fil)
    filx = get(filtup, 1, 1)
    fily = get(filtup, 2, 1)
    filc = get(filtup, 3, 1)
    filp = get(filtup, 4, 1)

    imgtup = size(img)
    x_out = get(imgtup, 1, 1)
    y_out = get(imgtup, 2, 1)
    imgc = get(imgtup, 3, 1)

    bias = zeros(size(fil,3))


    # Gradient w.r.t. img uses convolution with flipped filter
    # filter is flipped in the convolution loop using different order of indices for the filter
    dl_dimg = convolve_multi(img, fil, bias, fil_idx = (maxidx, idx) -> maxidx - idx + 1, for_grad=true)

   

    return dl_dimg 
end

function dl_dfil_calc(dl_dimg, fil; stride=1)
    # Gradient w.r.t. filter
    dl_dfil = zeros(size(fil))

    filtup = size(fil)
    filx = get(filtup, 1, 1)
    fily = get(filtup, 2, 1)
    filc = get(filtup, 3, 1)
    filp = get(filtup, 4, 1)

    imgtup = size(dl_dimg)
    x_out = get(imgtup, 1, 1)
    y_out = get(imgtup, 2, 1)
    imgc = get(imgtup, 3, 1)

    for z = 1:filp
        for j = 1:y_out
            for i = 1:x_out
                for ic = 1:imgc, fj = 1:fily, fi = 1:filx
                    img_i = stride * (i-1) + fi
                    img_j = stride * (j-1) + fj
                    if img_i <= y_out && img_j <= x_out
                        dl_dfil[fi, fj, ic, z] += 
                            dl_dimg[i,j,z] * img[img_i, img_j, ic]
                    end
                end
            end
        end
    end
    return dl_dfil
end

function new_img_size(img, fil; stride = 1, padding=:same)
    # dimensions of the single plane convolution result
    if padding == :same
        allpads = padding_same(img, fil, stride=stride)
    elseif padding == :none
        allpads = (0,0,0,0)
    end
    imgx, imgy = size(img)
    filx, fily = size(fil)
    x_out = floor(Int, (imgx - filx + allpads[1] + allpads[2]) / stride) + 1
    y_out = floor(Int, (imgy - fily + allpads[3] + allpads[4]) / stride) + 1

    return(x_out, y_out, allpads)
end

function new_img_size(imgx, imgy, filx, fily; stride=1, padding=:same)
    # dimensions of the single plane convolution result
    if padding == :same
        allpads = padding_same(imgx, imgy, filx, fily, stride=stride)
    elseif padding == :none
        allpads = (0,0,0,0)
    end
    x_out = floor(Int, (imgx - filx + allpads[1] + allpads[2]) / stride) + 1
    y_out = floor(Int, (imgy - fily + allpads[3] + allpads[4]) / stride) + 1

    return(x_out, y_out, allpads)
end

# use the actual img and fil with custom pad values to calc output size
function new_img_size(img, fil, pad_r1, pad_r2, pad_c1, pad_c2; stride)
    imgx, imgy = size(img)
    filx, fily = size(fil)
    # x_out = floor(Int, (imgx - filx + pad_r1 + pad_r2) / stride) + 1
    # y_out = floor(Int, (imgy - fily + pad_c1 + pad_c2) / stride) + 1
    x_out = ceil(((imgx + pad_r1 + pad_r2) - filx + 1) / stride)
    y_out = ceil(((imgy + pad_r3 + pad_r4) - fily + 1) / stride)

    return(x_out, y_out, (pad_r1, pad_r2, pad_c1, pad_c2))
end

# use img and fil dimensions with custom pad values to calc output size
function new_img_size(img_x, img_y, fil_x, fil_y, pad_r1, pad_r2, pad_c1, pad_c2; stride)
    x_out = ceil(((imgx + pad_r1 + pad_r2) - filx + 1) / stride)
    y_out = ceil(((imgy + pad_r3 + pad_r4) - fily + 1) / stride)

    return(x_out, y_out, (pad_r1, pad_r2, pad_c1, pad_c2))
end

# TODO we need a new_img_size for pooling outputs

"""
    dopad(arr, padh; padval=0.0, padv=padh)

    arr is a 2d image, optionally with channels and examples.
    padr an integer for the number of border elements to pad symmetrically.
    padval is the value to pad with.  Default 0 is nearly always the one you'll use.
    if padc is supplied, then padding can by asymmetric: padr is the number of padded rows,
        and padc is the number of padded columns.

    Returns padded array with same number of dimensions as input.
"""
function dopad(arr::Array{T,2} where T<:Real, allpads::NTuple{4, Signed}; padval=0.0) 
    padval = convert(eltype(arr), padval)
    m,n = size(arr)
    ret = fill(padval, m+allpads[1]+allpads[2], n+allpads[3]+allpads[4])
    for (jdx,j) = enumerate(allpads[3]+1:n+allpads[4])
        for (idx,i) = enumerate(allpads[1]+1:m+allpads[2])
            ret[i,j] = arr[idx,jdx]
        end
    end
    return ret
end

function dopad(arr::Array{T,2} where T<:Real, pad; padval=0.0)
    dopad(arr, (pad, pad, pad, pad), padval = padval)
end

# for dimensionality 3 tensor of type T
function dopad(arr::Array{T,3} where T<:Real, allpads::NTuple{4, Signed}; padval=0.0)
    padval = convert(eltype(arr), padval)
    m,n,c = size(arr)
    ret = fill(padval, m+allpads[1]+allpads[2], n+allpads[3]+allpads[4], c)
    # @show size(ret)
    for cdx = 1:c
        for (jdx,j) = enumerate(allpads[3]+1:n+allpads[4])
            for (idx,i) = enumerate(allpads[1]+1:m+allpads[2])
                ret[i, j, cdx] = arr[idx, jdx, cdx]
            end
        end
    end 
    return ret
end

function dopad(arr::Array{T,3} where T<:Real, pad; padval=0.0)
    dopad(arr, (pad, pad, pad, pad), padval = padval)
end

# for dimensionality 3 tensor of type T, with pre-allocated result
function dopad!(prepad::Array{T,4} where T<:Real, arr::Array{T,4} where T<:Real, allpads::NTuple{4, Signed})
    m,n,c = size(arr)
    for cdx = 1:c
        for (jdx,j) = enumerate(allpads[3]+1:n+allpads[4])
            for (idx,i) = enumerate(allpads[1]+1:m+allpads[2])
                prepad[i, j, cdx, xdx] = arr[idx, jdx, cdx, xdx]
            end
        end
    end   
end


# for dimensionality 4 tensor of type T
function dopad(arr::Array{T,4} where T<:Real, allpads::NTuple{4, Signed}; padval=0.0)
    padval = convert(eltype(arr), padval)
    m,n,c,x = size(arr)
    ret = fill(padval, padval, m+allpads[1]+allpads[2], n+allpads[3]+allpads[4], c, x)
    for xdx = 1:x
        for cdx = 1:c
            for (jdx,j) = enumerate(padc+1:n+padc)
                for (idx,i) = enumerate(padr+1:m+padr)
                    ret[i, j, cdx, xdx] = arr[idx, jdx, cdx, xdx]
                end
            end
        end   
    end
    return ret
end


# for dimensionality 4 tensor of type T, with pre-allocated result
function dopad!(prepad::Array{T,4} where T<:Real, arr::Array{T,4} where T<:Real, allpads::NTuple{4, Signed})
    m,n,c,x = size(arr)
    for xdx = 1:x
        for cdx = 1:c
            for (jdx,j) = enumerate(allpads[3]+1:n+allpads[4])
                for (idx,i) = enumerate(allpads[1]+1:m+allpads[2])
                    prepad[i, j, cdx, xdx] = arr[idx, jdx, cdx, xdx]
                end
            end
        end   
    end
end

function padding_same(img, fil; stride=1)
    imgx, imgy = size(img)
    filx, fily = size(fil)

    p_row = stride * imgx - imgx + filx - 1 
    p_r1 = ceil(Int, p_row/2) # p1 = p2 if imgsize is odd
    p_r2 = p_row - p_r1

    p_col = stride * imgy - imgy + fily - 1 
    p_c1 = ceil(Int, p_col/2)
    p_c2 = p_col - p_c1
    return p_r1, p_r2, p_c1, p_c2
end

function padding_same(imgx, imgy, filx, fily; stride=1)
    p_row = stride * imgx - imgx + filx - 1 
    p_r1 = ceil(Int, p_row/2) # p1 = p2 if imgsize is odd
    p_r2 = p_row - p_r1

    p_col = stride * imgy - imgy + fily - 1 
    p_c1 = ceil(Int, p_col/2)
    p_c2 = p_col - p_c1
    return p_r1, p_r2, p_c1, p_c2
end



function rot180(x)
    return reverse(reverse(x, dims=1), dims=2)
end

function pooling(img; pooldims=(2,2), stridedims=(2,2), mode=:max)
    if mode==:max
        func = maximum  # func => pool function
    elseif mode == :mean
        func = mean
    else
        error("mode must be :max or :mean")
    end

    img_x,img_y = size(img,1), size(img,2)
    c = ndims(img) == 3 ? size(img, 3) : 1

    poolx,pooly = pooldims
    stridex, stridey = stridedims

    x_out = div(img_x - poolx, stridex) + 1
    y_out = div(img_y - pooly, stridey) + 1

    ret = zeros(x_out, y_out, c)
    locations = fill(CartesianIndex(0,0), x_out, y_out, c)
    for z = 1:c
        for j = zip(1:y_out, 1:stridey:img_y)
            for i = zip(1:x_out, 1:stridex:img_x)
                submatview = @view(img[i[2]:i[2]+poolx-1, j[2]:j[2]+pooly-1, z])  # view saves 15 to 20 percent
                val = func(submatview)  
                loc = argmax(submatview)
                (mode == :max) && (locations[i[1], j[1], z] = loc)
                ret[i[1], j[1], z] = val
            end
        end
    end

    return ret, locations #, loc
end


function unpooling(pooled_img, locations; prevdims, pooldims=(2,2), stridedims=(2,2), mode=:max)
    # expand the imgstack to the larger dimensions filled with zeros
    # put the max value into the right part of the pooling submatrix
    img_x, img_y = size(pooled_img)
    prev_x, prev_y, prevch = prevdims
    stride_x, stride_y = stridedims
    poolx, pooly = pooldims
    divisor_fac = 1.0 / prod(pooldims)  # used only for mode=:mean; faster to multiply than divide

    ret = zeros(prevdims)
    if mode == :max
        for z = 1:prevch
            for j = zip(1:img_y,1:stride_y:prev_y)
                for i = zip(1:img_x,1:stride_x:prev_x)
                    submatview = @view(ret[i[2]:i[2]+poolx-1, j[2]:j[2]+pooly-1, z])  # view saves 15 to 20 percent
                    submatview[locations[img_x, img_y, z]]=pooled_img[i[1], j[1],z]
                end
            end
        end
    elseif mode == :mean
        for z = 1:prevch
            for j = zip(1:img_y, 1:stride_y:prev_y)
                for i = zip(1:img_x, 1:stride_x:prev_x)
                    submatview = @view(ret[i[2]:i[2]+poolx-1, j[2]:j[2]+pooly-1, z])  # view saves 15 to 20 percent
                    submatview .= pooled_img[i[1], j[1], z] * divisor_fac
                    # submatview[locations[img_x, img_y, z]]=pooled_img[i[1], j[1],z]
                end
            end
        end

    end

    return ret
end




"""
function flatten_img(img)

    flatten a 2d or 3d image for a conv net to use as a fully connected layer.
    follow the convention that features are rows and examples are columns.
    each column of an image plane is stacked below the prior column.  the next
    plan (or channel) comes below the prior channel.
    The size of img must have 3 dimensions even if each image has only 1 channel.
"""
function flatten_img(imgstack)
    if ndims(imgstack) != 3
        error("imgstack must have 3 dimensions even if img is 2d (use 1 for number of channels)")
    end
    x,y,c = size(imgstack)  # m x n image with c channels => z images like this
    return reshape(imgstack, prod((x,y,c)))
end


"""
function stack_img(fc, imgdims)

    Convert a flattened image back to an image stack.
    imgdims must provide 3 integer values:  m (rows or height) x n (columns or width) x c (number of channels).
    c, number of channels, must be provided for 2D images:  just use 1.

    returns: an imagestack that is imgx x imgy x nchannels x k where k is the number of images (or examples)

    Note: this returns an array that is a view of the input array.
"""
function stack_img(fc, imgdims)
    if length(imgdims) != 3
        error("imgdims must contain 3 integer values")
    end

    if ndims(fc) != 2
        error("fc must a 2d array with rows for image data and a column for each image")
    end

    imgx, imgy, nchannels = imgdims
    fcx, k = size(fc)

    if fcx != prod(imgdims)
        error("number of rows--number of elements for each image--does not match the image dimensions")
    end

    return reshape(fc, imgx, imgy, nchannels, k)
end

function stack_img(fc, xdim::Int, ydim::Int, nchannels::Int)
    if ndims(fc) != 2
        error("fc must a 2d array with rows for image data and a column for each image")
    end

    fcx, k = size(fc)

    if fcx != prod((xdim, ydim, nchannels))
        error("number of rows--number of elements for each image--does not match the image dimensions")
    end

    return reshape(fc, xdim, ydim, nchannels, k)

end



"""
make a 3d array by copying the 2d array into the 3rd dimension
"""
function catn(array, n::Integer)
    if n < 1
        error("n must be an integer greater than equal to 1")
    end
    ret = array
    for i = 1:n-1
        ret = cat(ret, array, dims = 3)
    end
    return ret
end


function he_initialize(wgts::AbstractArray)
        n_in = sum(size(wgts))
        scale = 2.0 / n_in
        randn(size(wgts)) .* sqrt(scale)
end

function he_initialize(wgt_dims::Tuple)
        n_in = sum(wgt_dims)
        scale = 2.0 / n_in
        randn(wgt_dims) .* sqrt(scale)
end

function he_initialize(w...)
    he_initialize(w)
end