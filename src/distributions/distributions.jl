
abstract type Distribution{T} <: AbstractModelNode end

include("poisson.jl")
include("mvnormal.jl")
include("categorical.jl")
include("geometric.jl")
include("zipoisson.jl")

####
#	Functions for making the library compatible with Mill
####

logpdf(d::Distribution, x::Mill.ArrayNode) = logpdf(d, x.data)

####
#	Functions for making the library compatible with HierarchicalUtils
####

HierarchicalUtils.nodeshow(io::IO, ::T) where {T<:Distribution} = print(io, "$(nameof(T))")
HierarchicalUtils.NodeType(::Type{<:Distribution}) = LeafNode()
