
abstract type _Distribution{T} <: AbstractModelNode end

###
#   concrete distributions import
###
include("poisson.jl")
include("mvnormal.jl")
include("categorical.jl")
include("geometric.jl")

###
#  compatibility with Mill
###
logpdf(d::_Distribution, x::Mill.ArrayNode) = logpdf(d, x.data)

####
#  compatibility with HierarchicalUtils
####
HierarchicalUtils.nodeshow(io::IO, ::T) where {T<:_Distribution} = print(io, "$(nameof(T))")
HierarchicalUtils.NodeType(::Type{<:_Distribution}) = LeafNode()
