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
struct ProductNode{T<:Union{Tuple, NamedTuple}, U<:NTuple{N, UnitRange{Int}} where N} <: AbstractModelNode
    components::T
    dimensions::U
end

Flux.@functor ProductNode
Flux.trainable(m::ProductNode) = (m.components,)

Base.length(m::ProductNode) = m.dimensions[end].stop
Base.getindex(m::ProductNode, i...) = getindex(m.components, i...)
Base.haskey(m::ProductNode{<:NamedTuple}, k::Symbol) = haskey(m.components, k)

"""
    ProductNode(ps::Union{Tuple, NamedTuple})

Construct ProductNode with `ps` independent random variables. `ps` should be 
iterable (`Tuple` or `NamedTuple`) of one or more `AbstractModelNode`s.

# Examples
```jldoctest
julia> ProductNode(_MvNormal(3), _Categotical(10))
ProductNode
  ├── _MvNormal
  ╰── _Categorical
julia> ProductNode(a=_MvNormal(3), b=_Categorical(10))
ProductNode
  ├── a: _MvNormal
  ╰── b: _Categorical
```

"""
function ProductNode(ps::Union{Tuple, NamedTuple})
    dimensions = Vector{UnitRange{Int}}(undef, length(ps))
    start = 1
    for (i, p) in enumerate(ps)
        l = length(p)
        dimensions[i] = start:start + l - 1
        start += l
    end
    ProductNode(ps, tuple(dimensions...))
end
ProductNode(ms...) = ProductNode(ms)
ProductNode(;ms...) = ProductNode(NamedTuple(ms))

####
#	Functions for calculating full likelihood
####


function logpdf(m::ProductNode{<:Tuple}, x::AbstractMatrix{<:Real})
    mapreduce((c, d)->logpdf(c, x[d, :]), +, m.components, m.dimensions)
end
logpdf(m::ProductNode, x::Mill.ArrayNode) = logpdf(m, x.data)

function logpdf(m::ProductNode{<:Tuple}, x::Mill.ProductNode{<:Tuple})
    mapreduce((c, i)->logpdf(c, x.data[i]), +, m.components, 1:length(m.components))
end

function logpdf(m::ProductNode{<:NamedTuple{KM}}, x::Mill.ProductNode{<:NamedTuple{KD}}) where {KM, KD}
    mapreduce(k->logpdf(m.components[k], x.data[k]), +, KM)
end

####
#	Functions for sampling the model
####
Base.rand(m::ProductNode) = vcat([rand(p) for p in m.components]...)
Base.rand(m::ProductNode, n::Int) = vcat([reshape(rand(p, n), length(p), n) for p in m.components]...)

# TODO: add sampling for product node of type Mill.ProductNode

####
#	Functions for making the library compatible with HierarchicalUtils
####
HierarchicalUtils.NodeType(::Type{<:ProductNode}) = InnerNode()
HierarchicalUtils.nodeshow(io::IO, ::ProductNode) = print(io, "ProductNode")
HierarchicalUtils.printchildren(node::ProductNode) = node.components
