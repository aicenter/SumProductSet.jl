using Mill
using Mill: OneHotArray, MaybeHotArray, OneHotMatrix
using Random

const Maybe{T} = Union{T, Missing}
const RAND_STR_LEN = 25

make_missing(x::Mill.ArrayNode,   r::Real) = Mill.ArrayNode(make_missing(x.data, r), x.metadata)
make_missing(x::Mill.BagNode,     r::Real) = Mill.BagNode(make_missing(x.data, r), x.bags, x.metadata)
make_missing(x::Mill.ProductNode, r::Real) = Mill.ProductNode(map(x->make_missing(x, r), x.data), x.metadata)

make_missing(o::Array{T, N},          n::Int, r) where {T, N} = (selectdim(o, N, rand(1:n, clamp(round(Int, n*r), 0, n))) .= missing; o)
make_missing(x::Array{T, N}, t::Type, n::Int, r) where {T, N} = make_missing(Array{Maybe{t}, N}(x), n, r)

make_missing(x::OneHotArray{T},   r) where {T<:Integer}                              = MaybeHotMatrix(make_missing(x.indices, T, size(x, 2), r), size(x, 1))
make_missing(x::MaybeHotArray{T}, r) where {T<:Maybe{Integer}}                       = MaybeHotMatrix(make_missing(x.I,       T, size(x, 2), r), size(x, 1))
make_missing(x::Array{T},         r) where {T<:Union{U, Maybe{U}}} where {U<:Real}   =                make_missing(x,         T, size(x, 2), r)
make_missing(x::NGramMatrix{T},   r) where {T<:Union{U, Maybe{U}}} where {U<:String} = NGramMatrix(   make_missing(x.S,       T, size(x, 2), r))


make_uniform(x::Mill.ArrayNode)   = Mill.ArrayNode(make_uniform(x.data), x.metadata)
make_uniform(x::Mill.BagNode)     = Mill.BagNode(make_uniform(x.data), x.bags, x.metadata)
make_uniform(x::Mill.ProductNode) = Mill.ProductNode(map(x->make_uniform(x), x.data), x.metadata)

make_uniform(x::OneHotArray{T})   where {T<:Integer}                              = OneHotMatrix(  rand(1:size(x, 1), size(x, 2)), size(x, 1))
make_uniform(x::MaybeHotArray{T}) where {T<:Maybe{Integer}}                       = MaybeHotMatrix(rand(1:size(x, 1), size(x, 2)), size(x, 1))
make_uniform(x::Array{T})         where {T<:Union{U, Maybe{U}}} where {U<:Real}   = rand(U, size(x)...)
make_uniform(x::NGramMatrix{T})   where {T<:Union{U, Maybe{U}}} where {U<:String} = NGramMatrix(map(_->randstring(rand(1:RAND_STR_LEN)), 1:size(x, 2)))
