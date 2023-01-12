"""
struct ProductNode
    components::T
    dimensions::U
end

    ProductNode implements a product of independent random variables. Each random
    variable(s) can be of any type, which implements the interface of `Distributions`
    package (`logpdf` and `length`). Recall that `length` in case of distributions is
    the dimension of a samples.
"""
# TODO: rewrite docstring
struct ProductNode{T<:Union{Tuple, NamedTuple}, U<:NTuple{N, UnitRange{Int}} where N}
    components::T
    dimensions::U
end

Flux.@functor ProductNode
Flux.trainable(m::ProductNode) = (m.components,)
Base.length(m::ProductNode) = m.dimensions[end].stop
Base.getindex(m::ProductNode, i...) = getindex(m.components, i...)

"""
    ProductNode(ps::Tuple)

    ProductNode with `ps` independent random variables. Each random variable has to
    implement `logpdf` and `length`.
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

####
#	Functions for calculating full likelihood
####
function logpdf(m::ProductNode{T}, x::AbstractMatrix{U}) where {T<:Tuple, U<:Real}
    mapreduce((c, d)->logpdf(c, x[d, :]), +, m.components, m.dimensions)
end
logpdf(m::ProductNode, x::Mill.ArrayNode) = logpdf(m, x.data)

function logpdf(m::ProductNode{T}, x::Mill.ProductNode{U}) where {T<:Tuple, U<:Tuple}
    mapreduce((c, i)->logpdf(c, x.data[i]), +, m.components, 1:length(m.components))
end

function logpdf(m::ProductNode{<:NamedTuple{KM}}, x::Mill.ProductNode{<:NamedTuple{KD}}) where {KM, KD}
    # TODO: add check for KM and KD
    mapreduce(k->logpdf(m.components[k], x.data[k]), +, KM)
end

####
#	Functions for sampling the model
####
Base.rand(m::ProductNode) = vcat([rand(p) for p in m.components]...)
# TODO: remove reshape
Base.rand(m::ProductNode, n::Int) = vcat([reshape(rand(p, n), length(p), n) for p in m.components]...)

# TODO: add sampling for product node of type Mill.ProductNode

####
#	Functions for making the library compatible with HierarchicalUtils
####
HierarchicalUtils.NodeType(::Type{<:ProductNode}) = InnerNode()
HierarchicalUtils.nodeshow(io::IO, ::ProductNode) = print(io, "ProductNode")
HierarchicalUtils.printchildren(node::ProductNode) = node.components
