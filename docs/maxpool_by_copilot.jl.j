function maxpool(x, pool_size; return_mask=false)
    pool_h, pool_w = pool_size
    out_h = div(size(x, 1), pool_h)
    out_w = div(size(x, 2), pool_w)
    out_c = size(x, 3)
    out_n = size(x, 4)
    
    pooled = zeros(Float32, out_h, out_w, out_c, out_n)
    mask = falses(size(x))
    
    for n in 1:out_n
        for c in 1:out_c
            for i in 1:out_h
                for j in 1:out_w
                    h_start = (i - 1) * pool_h + 1
                    h_end = i * pool_h
                    w_start = (j - 1) * pool_w + 1
                    w_end = j * pool_w
                    
                    window = x[h_start:h_end, w_start:w_end, c, n]
                    max_val = maximum(window)
                    pooled[i, j, c, n] = max_val
                    
                    if return_mask
                        mask[h_start:h_end, w_start:w_end, c, n] .= window .== max_val
                    end
                end
            end
        end
    end
    
    if return_mask
        return pooled, mask
    else
        return pooled
    end
end