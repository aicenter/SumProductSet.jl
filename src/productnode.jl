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
struct ProductNode{T<:Tuple,U<:NTuple{N,UnitRange{Int}} where N}
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
function ProductNode(ps::Tuple)
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
function logpdf(m::ProductNode, x::AbstractMatrix)
    o = logpdf(m.components[1], x[m.dimensions[1],:])
    for i in 2:length(m.components)
        o += logpdf(m.components[i], x[m.dimensions[i],:])
    end
    o
end

function logpdf(m::ProductNode, x::Mill.ArrayNode)
    o = logpdf(m.components[1], x.data[m.dimensions[1],:])
    for i in 2:length(m.components)
        o += logpdf(m.components[i], x.data[m.dimensions[i],:])
    end
    o
end

# each entry of PN has to have its own model distribution!!
function logpdf(m::ProductNode, x::Mill.ProductNode)
    o = logpdf(m.components[1], x.data[1])
    for i in 2:length(m.components)
        o += logpdf(m.components[i], x.data[i])
    end
    o
end

####
#	Functions for sampling the model
####
Base.rand(m::ProductNode) = vcat([rand(p) for p in m.components]...)

# TODO: add sampling for product node of type Mill.ProductNode

####
#	Functions for making the library compatible with HierarchicalUtils
####
HierarchicalUtils.NodeType(::Type{<:ProductNode}) = InnerNode()
HierarchicalUtils.nodeshow(io::IO, ::ProductNode) = print(io, "ProductNode")
HierarchicalUtils.printchildren(node::ProductNode) = node.components