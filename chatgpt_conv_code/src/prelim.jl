
# this works and added to sample_code.jl
function convlayerspec(;name::Symbol, activation::Symbol,adj::Float64=0.002, h::Int64, w::Int64, outch::Int64, f_h::Int64, f_w::Int64, inch::Int64, pad::Symbol=:same)
    LayerSpec(name=:conv1, kind=:conv, activation=activation, adj=adj, h=h, w=w, outch=outch,f_h=f_h, f_w=f_w, inch=inch, pad=pad )
end