"""
    SetNode{F<:AbstractModelNode, C<:AbstractModelNode} <: AbstractModelNode

Implement IID cluster model for random finite sets as `AbstractModelNode` given
feature and cardinality `AbstractModelNode`s.  

# Examples
```jldoctest
julia> Random.seed!(0);
julia> m = SetNode(MvNormal(3), Poisson())
SetNode
  ├── c: Poisson
  ╰── f: MvNormal
julia> x = rand(m, 4)
BagNode  # 4 obs, 128 bytes
  ╰── ArrayNode(3×16 Array with Float64 elements)  # 16 obs, 432 bytes
julia> logpdf(m, x)
1×4 Matrix{Float64}:
 -21.7319  -9.39979  -20.53  -7.74069
```
"""
struct SetNode{F<:AbstractModelNode, C<:AbstractModelNode} <: AbstractModelNode
    feature::F
    cardinality::C
end

Flux.@functor SetNode

####
#   Functions for calculating the likelihood
####

function logpdf(m::SetNode, x::Mill.BagNode)
    l = logpdf(m.feature, x.data)
    mapreduce(b->logpdf(m.cardinality, length(b)) .+ sum(l[b]) .+ logfactorial(length(b)), hcat, x.bags.bags)
end

####
    #   Functions for generating random samples
####

function Base.rand(m::SetNode, n::Int)
    if n == 0
        Mill.BagNode(missing, [1:0])
    else
        Mill.catobs(map(_->rand(m), 1:n))
    end
end
function Base.rand(m::SetNode)
    n = only(rand(m.cardinality).data)
    if n == 0
        Mill.BagNode(missing, [1:0])
    else
        Mill.BagNode(rand(m.feature, n), [1:n])
    end
end

####
#   Functions for making the library compatible with HierarchicalUtils
####

HierarchicalUtils.NodeType(::Type{<:SetNode}) = InnerNode()
HierarchicalUtils.nodeshow(io::IO, ::SetNode) = print(io, "SetNode")
HierarchicalUtils.printchildren(m::SetNode) = (c=m.cardinality, f=m.feature)

####
#   Functions for making the library compatible with Base
####

Base.length(m::SetNode) = length(m.feature)
