# simple_conv.jl test script

# %%
using Revise
Revise.revise()

# %%
cd("/users/lewis/Code/nn by hand")
pwd()

# %%
includet("../src/simple_conv.jl")
# Revise.track("../src/simple_conv.jl")

# %%

edg
# %%

convolve_multi(edg, double_fil)