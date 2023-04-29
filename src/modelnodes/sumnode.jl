"""
    SumNode{T<:Real, C<:AbstractModelNode} <: AbstractModelNode

Implement a mixture of identically distributed `AbstractModelNode`s as `AbstractModelNode`.

# Examples
```jldoctest
julia> Random.seed!(0);
julia> m = SumNode(_MvNormal(3), _MvNormal(3))
SumNode
  ├── _MvNormal
  ╰── _MvNormal
julia> x = rand(m, 5)
3×5 Matrix{Float64}:
 -2.24964   -1.77245  0.103564  -0.213714  -0.751201
 -0.549852   2.14755  0.384004  -2.07396   -0.672539
  0.244851  -1.82564  0.295006   0.145823  -1.79069
julia> logpdf(m, x)
5-element Vector{Float64}:
 -4.8343462870033225
 -6.118255930196227
 -3.712008533395728
 -4.277725228925475
 -4.3782586487431745
```
"""
struct SumNode{T<:Real, C<:AbstractModelNode} <: AbstractModelNode
    components::Vector{C}
    prior::Vector{T}
end

"""
    SumNode(components::Vector, prior::Vector)
    SumNode(components::Vector; dtype::Type{<:Real}) 

    Mixture of components. Each component has to be a valid pdf. If prior vector 
    is not provided, it is initialized uniformly.

    # Arguments
    - `components::Vector{C}`: Vector of components of same type.
    - `prior::Vector{T}`: Vector of log sumnode-weights, there is one weight for every component.
    - `dtype::Type{<:Real}` : Data type which initialized `prior` should have.
"""
function SumNode(components::Vector; dtype::Type{<:Real}=Float64) 
    n = length(components); 
    SumNode(components, ones(dtype, n))
end
SumNode(ms::AbstractModelNode...) = SumNode(collect(ms))

Base.getindex(m::SumNode, i::Int) = (c = m.components[i], p = m.prior[i])
Base.length(m::SumNode) = length(m.components[1])

Flux.@functor SumNode

"""
    logjnt(node, x)

log-jointlikelihood log p(x, y) of samples `x` and class/cluster labels `y` of a model `node`
"""
function logjnt(m::SumNode, x::Union{AbstractMatrix, Mill.AbstractMillNode})
    logcmps = transpose(mapreduce(c->logpdf(c, x), hcat, m.components))
    logw = logsoftmax(m.prior)
    logcmps .+ logw
end

"""
    logpdf(node, x)

log-likelihood of samples `x` of a model `node`
"""
function logpdf(m::SumNode, x::Union{AbstractMatrix, Mill.AbstractMillNode})
    vec(logsumexp(logjnt(m, x), dims=1))
end

####
#	Functions for sampling the model
####

_sampleids(m::SumNode, n::Int) = sample(1:length(m.prior), Weights(softmax(m.prior)), n)
_sampleids(m::SumNode) = _sampleids(m, 1)[]

function Base.rand(m::SumNode, n::Int)
    if n == 0 
        return zeros(eltype(m.prior), length(m), 0)
    else
        return hcat(rand.(m.components[_sampleids(m, n)])...)
    end
end
Base.rand(m::SumNode) = rand(m, 1)

function Base.rand(m::SumNode{<:Real, <:SetNode}, n::Int)
    if n == 0 
        return missing
    else
        return Mill.catobs(rand.(m.components[_sampleids(m, n)])...)
    end
end
# Base.rand(m::SumNode{T, C} where C<:SetNode) where T<:Real = rand(m.components[_sampleids(m)])
Base.rand(m::SumNode{<:Real, <:SetNode}) = rand(m.components[_sampleids(m)])

function randwithlabel(m::SumNode, n::Int)
    ids = _sampleids(m, n)
    x = hcat(rand.(m.components[ids])...)
    x, ids
end
function randwithlabel(m::SumNode)
    xm, ids = randwithlabel(m, 1)
    xm, ids[]
end

function randwithlabel(m::SumNode{<:Real, <:SetNode}, n::Int)
    ids = _sampleids(m, n)
    x = Mill.catobs(rand.(m.components[ids])...)
    x, ids
end
function randwithlabel(m::SumNode{<:Real, <:SetNode})
    xm, ids = randwithlabel(m, 1)
    xm, ids[]
end

####
#	Functions for making the library compatible with HierarchicalUtils
####
HierarchicalUtils.nodeshow(io::IO, ::SumNode) = print(io, "SumNode")
HierarchicalUtils.NodeType(::Type{<:SumNode}) = InnerNode()
HierarchicalUtils.printchildren(node::SumNode) = tuple(node.components...)
