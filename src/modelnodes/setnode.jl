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
    bags = x.bags.bags
    logp_f = logpdf(m.feature, x.data)
    logp_c = SumProductSet.logpdf(m.cardinality, hcat(length.(bags)...))
    # logp_c = ones(eltype(logp_f), 1, length(bags))
    _logpdf_set(logp_f, logp_c, bags)
end

function _logpdf_set(logp_f, logp_c, bags)
    lb = copy(logp_c)
    @inbounds for (bi, b) in enumerate(bags)
        lb[bi] += sum(logp_f[b]) + logfactorial(length(b))
    end
    lb
end

function _logpdf_set_back(logp_f, logp_c, bags, Δy)
    Δlogp_f = zero(logp_f)
    @inbounds for (bi, b) in enumerate(bags)
        for i in b
            Δlogp_f[i] += Δy[bi]
        end
    end
    Δlogp_f, Δy, NoTangent()
end

function ChainRulesCore.rrule(::typeof(_logpdf_set), args...)
    _logpdf_set_pullback = Δy -> (NoTangent(), _logpdf_set_back(args..., Δy)...)
    _logpdf_set(args...), _logpdf_set_pullback
end

# function logpdf(m::SetNode, x::Mill.BagNode)
#     l = logpdf(m.feature, x.data)
#     mapreduce(b->logpdf(m.cardinality, length(b)) .+ sum(l[b]) .+ logfactorial(length(b)), hcat, x.bags.bags)
# end

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
