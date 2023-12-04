"""
    SumNode{T<:Real, C<:AbstractModelNode} <: AbstractModelNode

Implement a mixture of identically distributed `AbstractModelNode`s as `AbstractModelNode`.

# Examples
```jldoctest
julia> Random.seed!(0);
julia> m = SumNode(MvNormal(3), MvNormal(3))
SumNode
  ├── MvNormal
  ╰── MvNormal
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
    weights::Vector{T}
end

Flux.@functor SumNode

SumNode(c::Vector; dtype::Type{<:Real}=Float32) = SumNode(c, ones(dtype, length(c)))

####
#   Functions for calculating the likelihood
####

# logjnt(m::SumNode, x) = mapreduce((c, w)->logpdf(c, x) .+ w, vcat, m.components, logsoftmax(m.weights))
logjnt(m::SumNode, x) = reduce(vcat, map((c, w)->logpdf(c, x) .+ w, m.components, logsoftmax(m.weights)))
logpdf(m::SumNode, x) = logsumexp(logjnt(m, x), dims=1)

####
#   Functions for generating random samples
####

_samplelatent(m::SumNode, n::Int) = sample(1:length(m.weights), Weights(softmax(m.weights)), n)
_samplelatent(m::SumNode) = _samplelatent(m, 1)[]

Base.rand(m::SumNode, n::Int) = reduce(Mill.catobs, rand.(m.components[_samplelatent(m, n)]))
Base.rand(m::SumNode) = rand(m, 1)

function randwithlabel(m::SumNode, n::Int)
    z = _samplelatent(m, n)
    x = reduce(Mill.catobs, rand.(m.components[z]))
    x, z
end
function randwithlabel(m::SumNode)
    x, z = randwithlabel(m, 1)
    x, only(z)
end

####
#   Functions for making the library compatible with HierarchicalUtils
####

HierarchicalUtils.nodeshow(io::IO, ::SumNode) = print(io, "SumNode")
HierarchicalUtils.NodeType(::Type{<:SumNode}) = InnerNode()
HierarchicalUtils.printchildren(m::SumNode) = tuple(m.components...)

####
#   Utilities
####

# Base.getindex(m::SumNode, i::Int) = (c = m.components[i], p = m.weights[i])
Base.length(m::SumNode) = length(m.components[1])
