"""
    SetNode{F<:AbstractModelNode, C<:AbstractModelNode} <: AbstractModelNode

Implement IID cluster model for random finite sets as `AbstractModelNode` given
feature and cardinality `AbstractModelNode`s.  

# Examples
```jldoctest
julia> Random.seed!(0);
julia> m = SetNode(_MvNormal(3), _Poisson())
SetNode
  ├── c: _Poisson
  ╰── f: _MvNormal
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
    mapreduce(b->logpdf(m.cardinality, length(b)) .+ sum(lp_inst[b]; dims=1) .+ logfactorial(length(b)), vcat, x.bags)
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
        @error "sampled unknown datatype"
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
HierarchicalUtils.printchildren(node::SetNode) = (c=node.cardinality, f=node.feature)
