
struct SumNode{T,C}
	components::Vector{C}
	prior::Vector{T}
	function SumNode(components::Vector{C}, prior::Vector{T}) where {T,C}
		ls = length.(components)
		@assert all( ls .== ls[1])
		new{T,C}(components, prior)
	end
end

"""
	SumNode(components::Vector, prior::Vector)
	SumNode(components::Vector; dtype::Type{<:Real}) 

	Mixture of components. Each component has to be a valid pdf. If prior vector 
	is not provided, it is initialized uniformly.
"""
function SumNode(components::Vector; dtype::Type{<:Real}=Float64) 
	n = length(components); 
	SumNode(components, ones(dtype, n))
end

Base.getindex(m::SumNode, i ::Int) = (c = m.components[i], p = m.prior[i])
Base.length(m::SumNode) = length(m.components[1])

Flux.@functor SumNode

"""
	logjnt(node, x)

	log-jointlikelihood of samples `x` and class/cluster label of a model `node`
"""
function logjnt(m::SumNode, x::Union{AbstractMatrix, Mill.AbstractMillNode})
	lkl = transpose(hcat(map(c -> logpdf(c, x), m.components)...))
	w = logsoftmax(m.prior)
	w .+ lkl
end

"""
	logpdf(node, x)

	log-likelihood of samples `x` of a model `node`
"""
function Distributions.logpdf(m::SumNode, x::Union{AbstractMatrix, Mill.AbstractMillNode})
	logsumexp(logjnt(m, x), dims = 1)[:]
end

####
#	Functions for sampling the model
####

_sampleids(m::SumNode, n::Int) = sample(1:length(m.prior), Weights(softmax(m.prior)), n)
_sampleids(m::SumNode) = _sampleids(m, 1)[]

function Base.rand(m::SumNode, n::Int)
	if n == 0 
		# fix fixed Float64 type
		return zeros(Float64, length(m), 0)
	else
		return hcat(rand.(m.components[_sampleids(m, n)])...)
	end
end
Base.rand(m::SumNode) = vec(rand(m, 1))

function Base.rand(m::SumNode{T, <:SetNode}, n::Int) where T 
	if n == 0 
		return missing
	else
		return Mill.catobs(rand.(m.components[_sampleids(m, n)])...)
	end
end
Base.rand(m::SumNode{T, <:SetNode}) where T = rand(m.components[_sampleids(m)])

function randwithlabel(m::SumNode, n::Int)
	ids = _sampleids(m, n)
	x = hcat(rand.(m.components[ids])...)
	x, ids
end
function randwithlabel(m::SumNode)
	xm, ids = randwithlabel(m, 1)
	vec(xm), ids[]
end

function randwithlabel(m::SumNode{T, <:SetNode}, n::Int) where T
	ids = _sampleids(m, n)
	x = Mill.catobs(rand.(m.components[ids])...)
	x, ids
end
function randwithlabel(m::SumNode{T, <:SetNode}) where T
	xm, ids = randwithlabel(m, 1)
	vec(xm), ids[]
end

####
#	Functions for making the library compatible with HierarchicalUtils
####
HierarchicalUtils.nodeshow(io::IO, ::SumNode) = print(io, "SumNode")
HierarchicalUtils.NodeType(::Type{<:SumNode}) = InnerNode()
HierarchicalUtils.printchildren(node::SumNode) = tuple(node.components...)
