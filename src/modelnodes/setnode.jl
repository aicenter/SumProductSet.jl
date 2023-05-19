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
4-element Vector{Float64}:
-21.731904383021657
-9.399785480434305
-20.530028604626864
-7.740694026190765
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
    mapreduce(b->logpdf(m.cardinality, length(b)) .+ sum(l[b]; dims=1) .+ logfactorial(length(b)), hcat, x.bags)
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
    n = rand(m.cardinality)
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

# Base.length(m::SetNode) = length(m.feature)

# function Base.getproperty(m::SetNode, name::Symbol)
#     if name in fieldnames(SetNode)
#         getfield(m, name)
#     elseif name == :f
#         getfield(m, :feature)
#     elseif name == :c
#         getfield(m, :cardinality)
#     else
#         error("type SetNode has no field $name")
#     end
# end