#=
    Initially, do this without pre-allocation.  TODO

=#

using GeneralNN  
using Random
using Statistics

# include("conv_net_funcs.jl")
include("colconv.jl")

function load_data(matfname, norm_mode="minmax")
    train = GeneralNN.Model_data()  # train holds all the data and layer inputs/outputs
    test = GeneralNN.Model_data()
    train.inputs, train.targets, test.inputs, test.targets = GeneralNN.extract_data(matfname)

    # fix the funny thing that MAT file extraction does to the type of the arrays
    train_x = convert(Array{Float64,2}, train.inputs)
    train_y = convert(Array{Float64,2}, train.targets)

    test_x = convert(Array{Float64,2}, test.inputs)
    test_y = convert(Array{Float64,2}, test.targets)

    return train_x, train_y, test_x, test_y
end


function allocate_data()
end

function check_preds(mat)
    res = Bool[]
    for v in axes(mat,2)
        push!(res,any(i -> i > 0.0,mat[:,v]))
    end
    return res
end

function basic(matfname; norm_mode="none", padding=:same)

    Random.seed!(1)

    # load training data and test data (if any)
    train_x, train_y, test_x, test_y = load_data(matfname)

    # set some useful variables
    in_k,n = size(train_x)  # number of features in_k (rows) by no. of examples n (columns)
    out_k = size(train_y,1)  # number of output units
    dotest = size(test_x, 1) > 0  # there is testing data

    if norm_mode != "none"
        norm_factors = GeneralNN.normalize_inputs!(train_x, "minmax")
        dotest && normalize_inputs!(test_x, norm_factors, "minmax") 
    end

    println("size train_x ", size(train_x))
    println("size train_y ", size(train_y))
    println("size test_x ", size(test_x))
    println("size test_y ", size(test_y))


    # structure of convnet
        # layer 1: input layer of 20 x 20 x n examples
                    # the image is 20 x 20 for b&w image with no color channels
        # layer 2: conv result 20 x 20 by 12 channels = 5408 values x n examples
                    # bias = vector of 12 elements
                    # pad = :same
                    # stride = 1
                    # first filters are 3 x 3 by 12 filters = 120 Weights
                    # first relu is 5408 values
        # layer 3: second conv result 18 x 18 by 8 = 2592 values x n examples
                    # second filters are 3 x 3 by 8 filters =  72 Wgts
                    # bias = vector of 8 elements
                    # pad = 0
                    # stride = 1
                    # second relu is 2592 values
        # layer 4:  maxpooling output for 2 x 2 block 
                    # output is is 10 x 10 x n = 648 values (no Wgts)
        # layer 5: fc is 800 x 5000 => flatten imgstack to 648 x 5000; 
        # layer 6: affine from 648 to 324, relu activation
        # layer 7: affine from 300 to 100, relu activation
        # layer 8: affine from 100 to 10, relu activation
        #          output softmax => affine weight (10, 324)  # softmax is 10 x 5000



    # debug
    # println(join((x_out, y_out, pad), " "))
    # error("that's all folks...")


    # declare and pre-allocate image stacks, filter stacks and bias vectors

    println("\npre-allocate storage arrays, weights")

    # image size values
    filx = fily = 3
    imgx1 = imgy1 = Int(sqrt(in_k))   # only works because we know the images are square!
    a1 = reshape(train_x, imgx1, imgy1, 1, :)  # a1 is the input image reshaped to an image stack
    println("\nsize of input image stack: ", size(a1))

    # layer 2: first conv z2 is z2 and activation a2
    stride2 = 1
    inch = 1  # in channels--channels of input image
    filt_out2 = 12 # out channels--new channels for each image filter
    w2 = he_initialize((filx,fily, inch, filt_out2)) # w2 is the filter stack->the linear weights for the 2nd layer
    bias2 = fill(0.0,filt_out2)
    dw2 = similar(w2)
    dbias2 = similar(bias2)
    pad2 = 1
    
    layer2 = ConvLayer(w2, bias2, stride2, pad2)
    input_size2, output_size2 = conv_dims(layer2,a1)
    (; output_height, output_width, output_channels, batch_size) = output_size2
    @show input_size2, output_size2
    layer2conv = ConvInputs(
        zeros(filx * fily * inch, n * output_size2.height * output_size2.width), # x_col
        zeros(filx * fily * inch, filt_out2), # w_col
        zeros(filt_out2), # bias
        output_channels, # out_channels
        output_height, # output_height
        output_width # output_width
        )

    
    z2 = rand(height, width, channels, batch_size)
    @show size(z2)
    a2 = similar(z2)
    dz2 = similar(z2)
    eps2 = similar(z2)

    # layer 3: second conv z3=imgstack3 and a3
    filt_out3 = 8
    # imgx3, imgy3, _ = size(a2)  # from previous conv layer
    # @show imgx3, imgy3
    
    stride3 = 1
    pad3 = 1
    inch = filt_out2 # previous out
    w3 = he_initialize((filx,fily, inch, filt_out3))
    # w3 = reshape(w3,filx, fily, inch, filt_out3)
    bias3 = fill(0.0, filt_out3)

    layer3 = ConvLayer(w3, bias3, stride3, pad3)
    input_size3, output_size3 = conv_dims(layer3,a2)
    @show input_size3, output_size3
    (; output_height, output_width, output_channels, batch_size) = output_size3
    layer3conv = ConvInputs(
        zeros(filx * fily * inch, n * output_height * output_width), # x_col
        zeros(filx * fily * inch, filt_out3), # w_col
        zeros(filt_out3), # bias
        output_channels, # out_channels
        output_height, # output_height
        output_width # output_width
        )
    z3 = rand(height, width, channels, batch_size)
    @show size(z3)
    a3 = similar(z3)
    dz3 = similar(z3)
    eps3 = similar(z3)



    # layer 4
    stride4 = 2
    size4 = 2
    mode4 = :max
    layer4 = PoolLayer(size4, stride4, mode4)

    dims4 = pool_dims(layer4, a3)
    (; height, width, channels, batch_size) = dims4
    a4 = rand(height, width, channels, batch_size)
    locs4 = fill(CartesianIndex(0,0), height, width, channels, batch_size)
    dz4 = similar(a4)
    eps4 = similar(a4)


    # layer 5, fully connected (flattened): nothing to do here as this is same as layer 4
    z5 = zeros(height * width * channels, n)
    a5 = similar(z5)
    # dz5 = similar(z5)  # not sure we need this
    eps5 = similar(a5)
    @show size(eps5)


    # layer 6, fully connected, linear transform, relu activation
    w6 = he_initialize(300, 800)  # TODO replace constants
    bias6 = fill(0.0, 300)
    z6 = rand(300,n)
    a6 = similar(z6)

    eps6 = similar(a6)
    dz6 = similar(a6)   # ???
    dw6 = similar(w6)
    dbias6 = similar(bias6)
    # @show size(a6)

    # layer 7, fully connected, linear transform, relu activation
    w7 = he_initialize(100, 300)
    bias7 = fill(0.0, 100)
    z7 = rand(100, n)
    a7 = similar(z7)

    # grad7 = similar(z7)  # grad and dz might be different names for same thing?
    eps7 = similar(z7)   # eps (or epsilon) is called dA in some formulations
    dz7 = similar(z7)
    dw7 = similar(w7)
    dbias7 = similar(bias7)
    # @show size(a7)


    # layer 8, fully connected, softmax activation
    w8 = he_initialize(10,100)
    bias8 = fill(0.0, 10)
    z8 = rand(10,n)
    a8 = similar(z8)



    dw8 = similar(w8)
    dbias8 = similar(bias8)
    eps8 = similar(a8)
    # @show size(a8)



    ########################################################################
    #  feed forward
    ########################################################################

    println("\n ******** starting feedforward")

    # layer 2: first convolution
    (conv_inputs2, img_size2) = prep_forward(layer2, a1)
    (; x_col, w_col, bias, out_channels, batch_size, output_height, output_width) = conv_inputs2
    z2 .= forward_conv(x_col, w_col, bias, out_channels, batch_size, output_height, output_width)

    GeneralNN.relu!(a2, z2)
    @show size(a2) # (20, 20, 12, 5000)
    
    # layer 3: second convolution
    (conv_inputs3, img_size3) = prep_forward(layer3, a2)
    (; x_col, w_col, bias, out_channels, batch_size, output_height, output_width) = conv_inputs3
    z3[:] = forward_conv(x_col, w_col, bias, out_channels, batch_size, output_height, output_width)
    GeneralNN.relu!(a3, z3)

    @show size(a3)
    @show typeof(img_size3)
    @show typeof(conv_inputs3)

    # layer 4: max pooling
    # a4[:], locs4 = forward_pool(a3, layer4)
    forward_pool!(a4, locs4, a3, layer4)

    @show size(a4), size(locs4)


    # layer 5: flatten images to fully connected vector
    z5 = flatten_img(a4)
    a5 = similar(z5)
    # GeneralNN.relu!(a5, z5)  # pretty sure we don't want to do this...
    @show size(a5)
    

    # layer 6: fully connected
    GeneralNN.affine!(z6,a5, w6, bias6)
    GeneralNN.relu!(a6, z6)
    @show size(a6), size(w6), size(bias6)
    

    # layer 7
    GeneralNN.affine!(z7,a6, w7, bias7)
    GeneralNN.relu!(a7, z7)
    @show size(a7)


    # layer 8: softmax activation
    GeneralNN.affine!(z8, a7, w8, bias8)
    GeneralNN.softmax!(a8, z8)
    @show size(a8)
    @show a8[:, 1]

    println(sum(check_preds(a8)))

 
    ########################################################################
    #  back prop
    ########################################################################

    println("\n ******** starting backprop\n")

    # output layer
    GeneralNN.backprop_classify!(eps8, a8, train_y) # epsilon = preds .- targets
    dw8[:] = eps8 * a7'


    GeneralNN.backprop_weights!(dw8, dbias8, eps8, a7, n)

    @show size(eps8), size(dw8), size(dbias8)


    # layer 7
    GeneralNN.inbound_epsilon!(eps7, w8, eps8)
    GeneralNN.relu_gradient!(dz7,a7)
    GeneralNN.current_lr_epsilon!(eps7, dz7)
    GeneralNN.backprop_weights!(dw7, dbias7, eps7, a6, n)
    @show size(eps7), size(dz7), size(dw7), size(dbias7)

    

    # layer 6
    GeneralNN.inbound_epsilon!(eps6, w7, eps7)
    GeneralNN.relu_gradient!(dz6,a6)
    GeneralNN.current_lr_epsilon!(eps6, dz6)
    GeneralNN.backprop_weights!(dw6, dbias6, eps6, a5, n)
    @show size(eps6), size(dz6), size(dw6), size(dbias6)

    # layer 5
    GeneralNN.inbound_epsilon!(eps5, w6, eps6)
    @show size(eps5)


    # layer 4--there are no weights to take the derivative of
    eps4 = reshape(eps5, size(a4))
    @show size(eps4), "still pooled"   # this is still pooled
    dL_dx4 = backward_pool(layer4, eps4, size(a3),locs4)
    @show size(dL_dx4)
    
    # layer 3
    GeneralNN.relu_gradient!(dz3,a3)
    @show size(dL_dx4), size(dz3), size(a3)
    # dL_dx3, dL_dw3, dL_db3 = backward_conv(layer3, dL_dx4, img_size3, conv_inputs3)
    dL_dx3, dL_dw3, dL_db3 = backward_conv(layer3, dz3, img_size3, conv_inputs3)
    # GeneralNN.current_lr_epsilon!(dL_dx3,dz3)
    @show size(dL_dx3), size(dL_dw3), size(dL_db3)

    # layer 2
    GeneralNN.relu_gradient!(dz2, a2)
    dL_dx2, dL_dw3, dL_db3 = backward_conv(layer2, dz2, img_size2, conv_inputs2)
    
    #update weights
    # layer 8

    # layer 7

    # layer 6

    # layer 5

    # layer 4

    # layer 3

    # layer 2


    # return w2, a8, train_y
end

function do_the_rest() 

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
