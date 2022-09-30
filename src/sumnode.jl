
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
	SumNode(components::Vector) 

	Mixture of components. Each component has to be a valid pdf. If prior vector 
	is not provided, it is initialized randomly.
"""
function SumNode(components::Vector) 
	n = length(components); 
	SumNode(components, rand(Float64, n))
end

Base.getindex(m::SumNode, i ::Int) = (c = m.components[i], p = m.prior[i])
Base.length(m::SumNode) = length(m.components[1])

Flux.@functor SumNode


function logjnt(m::SumNode, x::Union{AbstractMatrix, Mill.AbstractMillNode})
	lkl = transpose(hcat(map(c -> logpdf(c, x) ,m.components)...))
	w = m.prior .- logsumexp(m.prior)
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
Base.rand(m::SumNode) = rand(m.components[sample(Weights(softmax(m.prior)))])

function randwithlabel(m::SumNode)
	component = sample(Weights(softmax(m.prior)))
	x = rand(m.components[component])
	x, component 
end

####
#	Functions for making the library compatible with HierarchicalUtils
####
HierarchicalUtils.nodeshow(io::IO, ::SumNode) = print(io, "SumNode")
HierarchicalUtils.NodeType(::Type{<:SumNode}) = InnerNode()
HierarchicalUtils.printchildren(node::SumNode) = tuple(node.components...)


####
#	Functions for comparibility with Mill.jl
####
