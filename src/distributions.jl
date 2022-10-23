###
#   concrete distributions import
###
include("distributions/poisson.jl")
include("distributions/mvnormal.jl")
include("distributions/categorical.jl")

###
#  compatibility with Mill
###
Distributions.logpdf(d::Distribution, x::Mill.ArrayNode) = logpdf(d, x.data)

####
#  compatibility with HierarchicalUtils
####
HierarchicalUtils.nodeshow(io::IO, ::T) where {T<:Distribution} = print(io, "$(T)")
HierarchicalUtils.NodeType(::Type{<:Distribution}) = LeafNode()
