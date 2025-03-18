####################################################################
#  unrolling approach from a paper by MS Research
#     allows matrix multiplication to convolve
#     crazy fast compared to 2D geom approach
####################################################################
#   should we unroll the whole imgstack or just go one at a time?  depends on whether we can reuse
#   implement stride, padding, same for unrolled convolutions

function convolve_unroll_all(img, unfil, filx, fily; stri=1, pad=0, same=false)
    if ndims(img) == 3
        imgx, imgy, imgc = size(img)
    elseif ndims(img)== 2
        imgx, imgy = size(img)
        imgc = 1
    else
        error("Image slice must have 2 or 3 dimensions.")
    end

    (x_out, y_out, pad) = new_img_size(img, filx, fily; stride=stride, pad=pad, same=false)
    imgflat = zeros(x_out+2*pad, (y_out+2*pad)*filx*fily, imgc)

    imgflat[:] = unroll_img(img, filx, fily; stride=stride, pad=pad, same=same)
    return convolve_unroll(imgflat, unfil)
end

# method using img rather than dims of image
function unroll_img(img, fil; stri=1, pad=0, same=false)
    filx, fily = size(fil)
    unroll_img(img, filx, fily; stri=stri, pad=pad, same=same)
end

# method using the x and y dimensions of the filter
function unroll_img(img, filx, fily; stri=1, pad=0, same=false)

    if ndims(img) == 3
        imgx, imgy, imgc = size(img)
    elseif ndims(img)== 2
        imgx, imgy = size(img)
        imgc = 1
    else
        error("Image slice must have 2 or 3 dimensions.")
    end

    if pad > 0
        img = dopad(img,pad)
    end

    l_fil = filx * fily
    x_out, y_out, pad = new_img_size(img, filx, fily; stride=stride, pad=pad)
    imgflat = Array{eltype(img),3}(undef,x_out,y_out*filx*fily, imgc)

    # debug
    # println("imgflat ",size(imgflat))
    # println("img   ", size(img))

    for z = 1:imgc
        for i = 1:x_out 
            for j = 1:y_out 
                t = 0
                for m=i:i+filx-(max(2*pad+1,1))
                    for n=j:j+fily-(max(2*pad+1,1)) 
                        t += 1  # column displacement (across part of row) for result matrix
                        imgflat[i,(j-1)*l_fil+t, z] = img[m,n, z]  
                        # println(m," ", n, " ",x[m,n])
                    end
                end
            end
        end
    end

    return imgflat
end

# method with input for img
function unroll_fil(img, fil; stri=1, pad=0, same=false)

    if ndims(img) == 3
        imgx, imgy, imgc = size(img)
    elseif ndims(img)== 2
        imgx, imgy = size(img)
        imgc = 1
    else
        error("Image slice must have 2 or 3 dimensions.")
    end

    unroll_fil(imgx, imgy, imgc, fil; stri=stri, pad=pad, same=same)
end

# method with inputs for imgx, imgy, imgc
function unroll_fil(imgx, imgy, imgc, fil; stri=1, pad=0, same=false)

    # filc = filter channels must equal image channels; filp = filter planes--number of output channels
    if ndims(fil) == 2 # one filter, one image channel
        filx, fily = size(fil)
        filc = filp = 1
    elseif ndims(fil) == 3  # one filter with multiple image channels
        filx, fily, filc = size(fil)
        filp = 1
    elseif ndims(fil) == 4  # multiple filters and multiple image channels
        filx, fily, filc, filp = size(fil)  
    else
        error("wrong number of dimensions for filter: $(ndims(fil))")
    end

    !(filc == imgc) && error("Number of channels in image and filter do not match.")   

    l_fil = filx * fily
    x_out, y_out, pad = new_img_size(imgx, imgy, filx, fily; stride=1, pad=pad)
    fil = reshape(fil,filx,fily, filc, filp)      # TODO this will change the sender OK????

    flat = reshape(permutedims(fil,[2,1,3,4]), l_fil, filc, filp)
    unfil = zeros(eltype(fil), (y_out*filx*fily, x_out, filc, filp))  

    # debug
    # println("flat  ", size(flat))
    # println("unfil ", size(unfil))

    # for z = 1:filp
        for j = 1:x_out
            st = (j-1) * l_fil + 1
            fin = st + l_fil - 1
            unfil[st:fin,j,:,:] = flat
        end
    # end
    return unfil
end



function convolve_unroll(img, fil)
    if ndims(img) == 3
        imgx, imgy, imgc = size(img)
    elseif ndims(img)== 2
        imgx, imgy = size(img)
        imgc = 1
    else
        error("Image slice must have 2 or 3 dimensions.")
    end

    if ndims(fil) == 2 # one filter, one image channel
        filx, fily = size(fil)
        filc = filp = 1
    elseif ndims(fil) == 3  # one filter with multiple image channels
        filx, fily, filc = size(fil)
        filp = 1
    elseif ndims(fil) == 4  # multiple filters and multiple image channels
        filx, fily, filc, filp = size(fil)  
    else
        error("wrong number of dimensions for filter: $(ndims(fil))")
    end

    # this is wrong
    if !(filc == imgc) 
        error("Number of channels in image and filter do not match.")
    end    

    ret = zeros(imgx, fily, filp)
    for z = 1:filp
        for c = 1:imgc
            ret[:,:,z] .+= @view(img[:,:,c]) * @view(fil[:,:,c,z])
        end
    end

    return ret
end
