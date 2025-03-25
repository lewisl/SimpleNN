

using GeneralNN  # must be installed in LOAD_PATH
using Random
using Statistics

include("conv_net_funcs.jl")

function basic(matfname, norm_mode="minmax"; unroll=false, padding=:same)
    # create data containers
    train = GeneralNN.Model_data()  # train holds all the data and layer inputs/outputs
    test = GeneralNN.Model_data()
    Random.seed!(1)

    # load training data and test data (if any)
    train.inputs, train.targets, test.inputs, test.targets = GeneralNN.extract_data(matfname)

    # fix the funny thing that MAT file extraction does to the type of the arrays
    train_x = convert(Array{Float64,2}, train.inputs)
    train_y = convert(Array{Float64,2}, train.targets)
    test_x = convert(Array{Float64,2}, test.inputs)
    test_y = convert(Array{Float64,2}, test.targets)

    # set some useful variables
    in_k,n = size(train_x)  # number of features in_k (rows) by no. of examples n (columns)
    out_k = size(train_y,1)  # number of output units
    dotest = size(test_x, 1) > 0  # there is testing data

    norm_factors = GeneralNN.normalize_inputs!(train_x, "minmax")
    dotest && normalize_inputs!(test_x, norm_factors, "minmax") 

    # debug
    # return any(isnan.(train.inputs))



    # structure of convnet
        # layer 1: inputs are 400 x 5000 examples
                    # the image is 20 x 20
        # layer 2: first conv result 18 x 18 by 8 channels = 5408 values  # if we pad we get 20 x 20 output
                    # first filters are 3 x 3 by 8 filters = 72 Wgts
                    # first relu is same:   x 5000
        # layer 3: second conv result 16 x 16 by 12 output channels = 6912 values
                    # second filters are 3 x 3 x 8 channels by 12 filters =  864 Wgts
                    # second relu is same by 5000
                    # maxpooling output is 8 x 8 x 12 = 1728 values (no Wgts)
        # layer 4: fc is 240 x 5000 => flatten imgstack to 768 x 5000; affine weight (240, 768))
                    # third relu is 240 by 5000
        # layer 5: affine and softmax => affine weight (10, 2450)
                    # softmax is 10 x 5000

    

    # debug
    # println(join((x_out, y_out, pad), " "))
    # error("that's all folks...")

    ########################################################################
    #  feed forward
    ########################################################################

    # first conv z2 and a2
    # image size values
    filx = fily = 3
    imgx = imgy = Int(sqrt(in_k))
    imgstack = reshape(train_x, imgx, imgy, 1, :)
    a1 = imgstack # effectively, an alias--no allocation
    println("\nsize of image stack: ", size(imgstack))
    stride = 1
    inch = 1  # in channels--channels of input image
    filt_k = 8 # out channels--new channels for each image filter
    w2 = rand(filx,fily,inch,filt_k)
    bias2 = fill(0.3,filt_k)
    (x_out, y_out, pad) = new_img_size(imgstack, filx, fily; stri=stri, pad=pad, same=false)

    z2 = zeros(x_out,y_out,filt_k,n)  # preallocate for multiple epochs
    if !unroll
        @time for ci = 1:n     #size(imgstack,4)  # ci = current image
            z2[:,:,:, ci] = convolve_multi(imgstack[:,:,:,ci], w2; stride=stride)   
        end
    else  # more than 13 times faster than FFT style (stack)!
        println("first unroll and convolve")

        @time unfil = unroll_fil(imgstack[:,:,:,1], w2) # only need to do this once
        @time for ci = 1:n
            z2[:,:,:, ci] = convolve_unroll_all(imgstack[:,:,:,ci], unfil, filx, fily, stri=stri, pad=pad)    
        end
        println("convolved z2 using unroll: ", size(z2))
    end
    # add bias to each out channel
    for i = 1:filt_k
        z2[:,:,i,:] .+= bias2[i]
    end

    # TODO -- do we want to reuse memory or allocate more?
    # first relu
    println("first relu")
    a2 = copy(z2)
    GeneralNN.relu!(a2, z2)  
    println("type of relu output a2: ", typeof(a2))
    println("size of relu output a2: ", size(a2))

    # second conv z3 and a3
    # image size values
    filx = fily = 3
    imgx, imgy = size(a2)  # from previous conv layer
    stri = 1
    inch = filt_k # previous out
    filt_k = 12
    w3 = rand(filx,fily,inch,filt_k)
    bias3 = fill(0.3, filt_k)
    (x_out, y_out, pad) = new_img_size(z2, filx, fily; stri=stri, pad=0, same=false)

    z3 = zeros(x_out, y_out, filt_k, n)
    if !unroll
        for ci = 1:n
            z3[:,:,:, ci] .= convolve_multi(a2[:,:,:, ci], w3; stri=stri)
        end
    else
        println("\nsecond unroll and convolve")
        println("size z3: ", size(z2), " size w3: ", size(w3))
        @time unfil = unroll_fil(z2[:,:,:,1], w3)
        @time for ci = 1:n
            z3[:,:,:, ci] = convolve_unroll_all(a2[:,:,:,ci], unfil, filx, fily, stri=stri, pad=pad)    
        end
    end
    # add bias to each out channel
    @time for c = 1:filt_k
        z3[:,:,c,:] .+= bias3[c]
    end

    println("type of 2nd conv, layer 3 output: ", typeof(z3))
    println("size of 2nd conv, layer 3 output: ", size(z3))

    # second relu
    println("\nsecond relu")
    @time begin
        a3 = copy(z3)
        GeneralNN.relu!(a3, z3)
    end
    println("type of relu output a3: ", typeof(a3))
    println("size of relu output a3: ", size(a3))

    # maxpooling a3pool
    println("\nmax pooling")
    @time begin
        a3x, a3y, a3c, a3n = size(a3)
        a3pool = zeros(Int(a3x/2), Int(a3y/2), a3c, a3n)  # TODO need reliable way to set pool output size
        a3pool_loc = Array{CartesianIndex{2},4}(undef, Int(a3x/2), Int(a3y/2), a3c, a3n) 
        for i in 1:a3n
            a3pool[:,:,:,i], a3pool_loc[:,:,:,i] = maxpooling(a3[:,:,:,i])
        end
    end
    println("size maxpooling: ", size(a3pool))
    println("size maxpool loc: ", size(a3pool_loc))
    println("type of maxpool loc ", typeof(a3pool_loc))    

    # layer 4: fully connected and relu  z4 and a4
    println("\nfully connected and relu activation z4 and a4")
    in_k = prod(size(a3pool)[1:3])
    out_k = 240
    theta4 = rand(out_k, in_k)
    bias4 = fill(0.3, out_k)
    a4 = z4 = rand(out_k, n) # initialize and allocate
    @time begin
        GeneralNN.affine!(z4, flatten_img(a3pool), theta4, bias4)
        GeneralNN.relu!(a4, z4)
    end
    println("size of fc and relu a4: ", size(a4))

    # layer 5: fully connected and softmax    z5 and a5
    println("\nfully connected z5 and softmax output a5")
    in_k = out_k  # previous layer out
    out_k = 10
    theta5 = rand(out_k, in_k)
    bias5 = fill(0.3, out_k)
    println("size theta4: ", size(theta5))
    a5 = z5 = rand(out_k, n) # initialize and allocate
    @time begin
        GeneralNN.affine!(z5, a4, theta5, bias5)
        GeneralNN.softmax!(a5, z5)
    end
    println("output size after linear and softmax: ", size(a5))

    ########################################################################
    #  back prop
    ########################################################################
        println("\n*****************")
        println("start of backprop")
        # output layer, layer 5--we only need epsilon, the difference
        dz5 = a5 .- train.targets  # called epsilon in FF nn's
        delta_th_5 = dz5 * a4'  # do we need 1/m -- we usually take the average as part of the weight update
        delta_b_5 = sum(dz5, dims=2) # ditto
        println("size of dz5: ", size(dz5))
        println("\nbackprop fully connected layer 5")
        println("size of delta_th_5: ", size(delta_th_5))
        println("size of delta_b_5: ", size(delta_b_5))
        println("size of dz5: ", size(dz5))

        # fully connected, layer 4
        println("\nbackprop fully connected layer 4")
        grad_a4 = zeros(size(a4))
        GeneralNN.relu_gradient!(grad_a4, a4)
        dz4 = theta5' * dz5 .* grad_a4
        delta_th_4 = dz4 * flatten_img(a3pool)'  # this seems weird
        delta_b_4 = sum(dz4, dims=2)

        println("size of delta_th_4: ", size(delta_th_4))
        println("size of delta_b_4: ", size(delta_b_4))
        println("size of dz4: ", size(dz4))

        # maxpooling
            # TODO we need to convert back to an imgstack.  we need to know the dimensions so we need a place
            #    to save dimensions per layer
            # unpooling
            #     max:  need a mask for where the max value is.  deriv is 1.  0 for the other values
            #     avg:  use 1/size of pooling grid times each value.
        println("\nbackprop max pooling")
        pre_unpool3 = theta4' * dz4
        un_pool_3 = zeros(a3x,a3y,a3c,a3n)
        imgstack = reshape(pre_unpool3, Int(a3x/2),Int(a3y/2),a3c,a3n)
        # @time for i = 1:a3n
        #     un_pool_3[:,:,:,i] = unpool(imgstack[:,:,:,i], a3pool_loc[:,:,:,i], mode="max")
        # end    # unpooling doesn't make any sense--has to be another backprop output to use at next lower layer
        # println("size of un_pool3:", size(un_pool_3))


        # 2nd conv
        # un_pool_3 is (16,16,12,5000)   z2 is (18,18,8,5000)
        # w3 is (3,3,12)

        println("\nbackprop of layer 3: convolve and relu")
        println("size a3: ", size(a3), " size un_pool_3: ", size(un_pool_3), " size w3: ", size(w3))
        grad_relu_3 = zeros(size(a3))   # zeros(size(un_pool_3))
        GeneralNN.relu_gradient!(grad_relu_3, a3)
        println("size grad of relu at layer 3: ", size(grad_relu_3))

        dz3 = zeros(size(a3))
        dz3[:] = grad_relu_3 .* un_pool_3
        println("size dz3: $(size(dz3))")

        delta_th_3 = zeros(size(w3))
        @time for i = 1:n 
            delta_th_3[:,:,:,:] += convolve_grad_w(a2[:,:,:,i], dz3[:,:,:,i],  w3)  # middle term? un_pool_3[:,:,:,i],
        end
        delta_th_3[:] = (1/n) .* delta_th_3
        delta_b_3 = (1/n) .* sum(dz3 ,dims=(1,2,4))[:]   # alternative to sum? un_pool_3
        println("size delta_th_3: $(size(delta_th_3)) size delta_b_3: $(size(delta_b_3))")

        # 1st relu at layer 2
        println("\nbackprop of layer 2: convolve and relu")
        println("size a2: ", size(a2),  " size w2: ", size(w2))
        grad_relu_2 = zeros(size(a2))   # zeros(size(un_pool_3))
        GeneralNN.relu_gradient!(grad_relu_2, a2)
        println("size grad of relu at layer 2: ", size(grad_relu_2))

        dz2 = zeros(size(a2))
        @time for i = 1:n
            dz2[:,:,:, i] = convolve_grad_x(dopad(dz3[:,:,:,i],2), w3)  # alternative to pad? un_pool_3
        end
        println("initial size of dz3 before gradient: $(size(dz2))")
        dz2[:] = dz2 .* grad_relu_2
        println("size dz2: $(size(dz2))")

        delta_th_2 = zeros(size(w2))
        @time for i = 1:n 
            delta_th_2[:,:,:,:] += convolve_grad_w(a1[:,:,:,i], dz2[:,:,:,i], w2)
        end
        delta_th_2[:] = (1/n) .* delta_th_2
        delta_b_2 = (1/n) .* sum(dz2,dims=(1,2,4))[:]
        println("size delta_th_2: $(size(delta_th_2)) size delta_b_2: $(size(delta_b_2))")
            
    
    println("that's all folks!...")

end

function simpletest()
    # use an input that has 2 obvious edges; create target using an 2x2 edge kernel; can we learn the kernel?

    # create x
    x = ones(6,8)
    x[:,3:6] .= 0.0

    # kernel
    k = [1 -1]

    # target with perfect edge recognition
    y = convolve_multi(x,k)
    
    # initialize kernel
    w = [-0.12 0.3]

    # set data dims to h,w,c,n
    x = reshape(x,6,8,1,1)
    y = reshape(y,6,7,1,1)

    for i = 1:10
        
        # TODO do a training loop here...

    end
end
