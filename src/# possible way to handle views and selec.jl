# possible way to handle views and selector for the base class of desired fields

julia> mutable struct inner
       a::SubArray{Float32,2}
       b::SubArray{Float32,4}
       end

julia> struct outer
       views::inner
       end

julia> arr2 = rand(20,50);

julia> arr4 = rand(20,50,5,10);

julia> arr2 = rand(Float32,20,50);

julia> arr4 = rand(Float32,20,50,5,10);