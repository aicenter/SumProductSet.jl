
abstract type _Distribution{T} end

###
#   concrete distributions import
###
include("distributions/poisson.jl")
include("distributions/mvnormal.jl")
include("distributions/categorical.jl")

###
#  compatibility with Mill
###
logpdf(d::_Distribution, x::Mill.ArrayNode) = logpdf(d, x.data)

####
#  compatibility with HierarchicalUtils
####
HierarchicalUtils.nodeshow(io::IO, ::T) where {T<:_Distribution} = print(io, "$(T)")
HierarchicalUtils.NodeType(::Type{<:_Distribution}) = LeafNode()
