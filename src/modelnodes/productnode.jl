"""
ProductNode{C<:AbstractModelNode, D<:Union{UnitRange{Int}, Vector{T}, T} where {T<:Symbol}} <: AbstractModelNode

Implement a product of independent random variables as `AbstractModelNode`.

# Examples
```jldoctest
julia> Random.seed!(0);
julia> x = Mill.ProductNode(a=Mill.ArrayNode([0. 1; 2 3]), b=Mill.ArrayNode([4. 5; 6. 7]))
ProductNode  # 2 obs, 16 bytes
  ├── a: ArrayNode(2×2 Array with Float64 elements)  # 2 obs, 80 bytes
  ╰── b: ArrayNode(2×2 Array with Float64 elements)  # 2 obs, 80 bytes
julia> m = ProductNode(a=MvNormal(2), b=MvNormal(2))
ProductNode (:a, :b)
  ├── MvNormal
  ╰── MvNormal
julia> logpdf(m, x)
1×2 Matrix{Float64}:
 -36.468  -51.6233
```
"""
struct ProductNode{C<:AbstractModelNode, D<:Union{UnitRange{Int}, Vector{T}, T} where {T<:Symbol}} <: AbstractModelNode
    components::Vector{C}
    dimensions::Vector{D}
end

Flux.@functor ProductNode
Flux.trainable(m::ProductNode) = (m.components,)

ProductNode(c::AbstractModelNode, d::Union{UnitRange{Int}, Vector{T}, T}) where {T<:Symbol} = ProductNode([c], [d]) # best to get rid of this in future
ProductNode(;ms...) = ProductNode(NamedTuple(ms))
ProductNode(ms::NamedTuple) = ProductNode(collect(values(ms)), collect(keys(ms)))

function ProductNode(components::Vector{<:AbstractModelNode})
  dimensions = Vector{UnitRange{Int}}(undef, length(components))
  start = 1
  for (i, c) in enumerate(components)
      l = length(c)
      dimensions[i] = start:start + l - 1
      start += l
  end
  ProductNode(components, collect(dimensions))
end

####
#	  Functions for calculating the likelihood
####

logpdf(m::ProductNode, x::AbstractMatrix)   = mapreduce((c, d)->logpdf(c, x[d, :]), +, m.components, m.dimensions)
logpdf(m::ProductNode, x::Mill.ProductNode) = mapreduce((c, d)->logpdf(c, x[d]),    +, m.components, m.dimensions)
logpdf(m::ProductNode, x::Mill.ArrayNode)   = logpdf(m, x.data)

####
#	  Functions for generating random samples
####

Base.rand(m::ProductNode{C, D}, n::Int) where {C<:AbstractModelNode, D<:Union{Vector{T}, T} where {T<:Symbol}} = 
  map((c, k)->k=>rand(c, n), m.components, m.dimensions) |> NamedTuple |> Mill.ProductNode
Base.rand(m::ProductNode{C, D}, n::Int) where {C<:AbstractModelNode, D<:UnitRange{Int}} = 
  Mill.ArrayNode(reduce(vcat, map(c->rand(c, n).data, m.components)))
  # mapreduce(c->rand(c, n), vcat, m.components)
  # reduce(sparse_vcat, map(c->rand(c, n), m.components))

Base.rand(m::ProductNode) = rand(m, 1)

####
#	  Functions for making the library compatible with HierarchicalUtils
####

HierarchicalUtils.NodeType(::Type{<:ProductNode}) = InnerNode()
HierarchicalUtils.nodeshow(io::IO, m::ProductNode) = (print(io, "ProductNode "), _print(io, m.dimensions))
HierarchicalUtils.printchildren(m::ProductNode) = m.components

_print(io, ::Vector{T}) where {T<:UnitRange{Int}} = print(io, "")
_print(io, x::Vector{T}) where {T<:Symbol} = print(io, "$(Tuple(x))")
_print(io, x::Vector{T}) where {T<:Vector{<:Symbol}} = foreach(d->print(io, "$(d) "), x)


####
#	  Utilities
####

Base.length(m::ProductNode) = sum(length, m.components)
