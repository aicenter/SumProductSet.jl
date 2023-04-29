"""
    ProductNode{T<:Union{Tuple, NamedTuple}, U<:NTuple{N, UnitRange{Int}} where N} <: AbstractModelNode

Implement a product of independent random variables as `AbstractModelNode`.

# Examples
```jldoctest
julia> Random.seed!(0);
julia> x = Mill.ProductNode(a=Mill.ArrayNode([0. 1; 2 3]), b=Mill.ArrayNode([4. 5; 6. 7]))
ProductNode  # 2 obs, 16 bytes
  ├── a: ArrayNode(2×2 Array with Float64 elements)  # 2 obs, 80 bytes
  ╰── b: ArrayNode(2×2 Array with Float64 elements)  # 2 obs, 80 bytes
julia> m = ProductNode(a=_MvNormal(2), b=_MvNormal(2))
ProductNode
  ├── a: _MvNormal
  ╰── b: _MvNormal
julia> logpdf(m, x)
2-element Vector{Float64}:
 -36.468016427033014
 -51.62330239036811
```
"""
struct ProductNode{C<:AbstractModelNode, D<:Union{UnitRange{Int}, Tuple{Vararg{T}}, T} where {T<:Symbol}} <: AbstractModelNode
    components::Vector{C}
    dimensions::Vector{D}
end

Flux.@functor ProductNode
Flux.trainable(m::ProductNode) = (m.components,)

ProductNode(c::AbstractModelNode, d::Union{UnitRange{Int}, Tuple{Vararg{T}}, T}) where {T<:Symbol} = ProductNode([c], [d]) # best to get rid of this in future

####
#	Functions for calculating full likelihood
####

logpdf(m::ProductNode, x::AbstractMatrix)   = mapreduce((c, d)->logpdf(c, x[d, :]), +, m.components, m.dimensions)
logpdf(m::ProductNode, x::Mill.ProductNode) = mapreduce((c, d)->logpdf(c, x[d]),    +, m.components, m.dimensions)

####
#	Functions for generating random samples
####

# Base.rand(m::ProductNode) = rand(m, 1)
# Base.rand(m::ProductNode, n::Int) = Mill.ProductNode(map((k, c)->k=>rand(c, n) |> _reshape, m.components, m.dimensions))

# _reshape(x::Vector{Int}) = reshape(x, 1, :)
# _reshape(x) = x

####
#	Functions for making the library compatible with HierarchicalUtils
####

HierarchicalUtils.NodeType(::Type{<:ProductNode}) = InnerNode()
HierarchicalUtils.nodeshow(io::IO, m::ProductNode) = (print(io, "ProductNode "), _print(io, m.dimensions))
HierarchicalUtils.printchildren(node::ProductNode) = node.components

_print(io, x::Vector{T}) where {T<:Symbol} = print(io, "$(Tuple(x))")
_print(io, x::Vector{T}) where {T<:Tuple{Vararg{<:Symbol}}} = foreach(d->print(io, "$(d) "), x)
