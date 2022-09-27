struct SetNode{T, S}
    feature::T
    cardinality::S
end

Flux.@functor SetNode

Base.length(m::SetNode) = length(m.feature)


function Distributions.logpdf(m::SetNode, x::Mill.BagNode)

    lp_inst = logpdf(m.feature, x.data)  # might not work on nonvector data
    lp_bag = mapreduce(b->logpdf(m.cardinality, length(b)) + sum(lp_inst[b]) + logfactorial(length(b)), vcat, x.bags)
    
    return lp_bag
end


####
#	Functions for sampling the model
####
_rand(m::SetNode) = rand(m.feature, rand(m.cardinality))
# Base.rand(m::SetNode) = Mill.BagNode(_rand(m), 2)
function Base.rand(m::SetNode)
    x = _rand(m)
    Mill.BagNode(Mill.ArrayNode(x), [1:size(x, 2)])
end

Base.rand(m::SetNode, n::Int) = Mill.catobs(map(()->rand(m), 1:n))

####
#	Functions for making the library compatible with HierarchicalUtils
####
HierarchicalUtils.NodeType(::Type{<:SetNode}) = InnerNode()
HierarchicalUtils.nodeshow(io::IO, ::SetNode) = print(io, "SetNode")
HierarchicalUtils.printchildren(node::SetNode) = [:f => node.feature, :c => node.cardinality]
