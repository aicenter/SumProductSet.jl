
"""
    Setnode(feature, cardinality)

    Iid cluster model of point process. 
    Both feature and cardinality has to be valid distributions/nodes with valid logpdf. 
"""
struct SetNode{T, S}
    feature::T
    cardinality::S
end

Flux.@functor SetNode

Base.length(m::SetNode) = length(m.feature)


function Base.getproperty(m::SetNode, name::Symbol)
    if name in fieldnames(SetNode)
        getfield(m, name)
    elseif name == :f
        getfield(m, :feature)
    elseif name == :c
        getfield(m, :cardinality)
    else
        error("type SetNode has no field $name")
    end
end


"""
    logpdf(node, x)

    log-likelihood of Mill bagnode `x` of a set model `node`
"""
function logpdf(m::SetNode, x::Mill.BagNode)
    lp_inst = logpdf(m.feature, x.data)
    mapreduce(b->logpdf(m.cardinality, length(b)) + sum(lp_inst[b]) + logfactorial(length(b)), vcat, x.bags)
end

####
#	Functions for sampling the model
####

function Base.rand(m::SetNode)
    card = rand(m.cardinality)
    x = rand(m.feature, card)
    if typeof(x) <: Matrix{<:Real}
        Mill.BagNode(x, [1:card])
    elseif typeof(x) <: Vector{<:Real}
        Mill.BagNode(hcat(x), [1:card])
    elseif typeof(x) <: Mill.AbstractMillNode
        Mill.BagNode(x, [1:card])
    else
        @error "sampled unknown dtype"
    end
end

function Base.rand(m::SetNode, n::Int)
    if n == 0
        Mill.BagNode(missing, [1:0])
    else
        Mill.catobs(map(_->rand(m), 1:n))
    end
end

####
#	Functions for making the library compatible with HierarchicalUtils
####
HierarchicalUtils.NodeType(::Type{<:SetNode}) = InnerNode()
HierarchicalUtils.nodeshow(io::IO, ::SetNode) = print(io, "SetNode")
HierarchicalUtils.printchildren(node::SetNode) = [:f => node.feature, :c => node.cardinality]
