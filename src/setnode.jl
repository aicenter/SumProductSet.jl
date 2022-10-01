
struct SetNode{T, S}
    feature::T
    cardinality::S
end

Flux.@functor SetNode

Base.length(m::SetNode) = length(m.feature)


function Distributions.logpdf(m::SetNode, x::Mill.BagNode)

    lp_inst = logpdf(m.feature, x.data)  # might not work on nonvector data
    mapreduce(b->logpdf(m.cardinality, length(b)) + sum(lp_inst[b]) + logfactorial(length(b)), vcat, x.bags)
end


####
#	Functions for sampling the model
####

function Base.rand(m::SetNode)
    card = rand(m.cardinality)
    x = rand(m.feature, card)
    if typeof(x) <: Matrix{<:Real}
        xm = Mill.ArrayNode(x)
    elseif typeof(x) <: Vector{<:Real}
        xm = Mill.ArrayModel(hcat(x))
    elseif typeof(x) <: Mill.ArrayNode
        xm = x
    elseif typeof(x) <: Mill.BagNode
        @error "rand not implemented yet for set of set"
    elseif typeof(x) <: Mill.ProductNode
        @error "rand not implemented yet for set of product nodes"
    end
    Mill.BagNode(xm, [1:card])
end

function Base.rand(m::SetNode, n::Int)
    if n == 0
        return Mill.BagNode(missing, [1:0])
    else
        return Mill.catobs(map(_->rand(m), 1:n))
    end
end

####
#	Functions for making the library compatible with HierarchicalUtils
####
HierarchicalUtils.NodeType(::Type{<:SetNode}) = InnerNode()
HierarchicalUtils.nodeshow(io::IO, ::SetNode) = print(io, "SetNode")
HierarchicalUtils.printchildren(node::SetNode) = [:f => node.feature, :c => node.cardinality]
