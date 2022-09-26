
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
	is not provided, it is initialized to uniform.
"""
function SumNode(components::Vector) 
	n = length(components); 
	SumNode(components, fill(1f0, n))
end

Base.getindex(m::SumNode, i ::Int) = (c = m.components[i], p = m.prior[i])
Base.length(m::SumNode) = length(m.components[1])

Flux.@functor SumNode

"""
	logpdf(node, x)

	log-likelihood of samples `x` of a model `node`
"""
function Distributions.logpdf(m::SumNode, x::AbstractMatrix)
	lkl = transpose(hcat(map(c -> logpdf(c, x) ,m.components)...))
	w = m.prior .- logsumexp(m.prior)
	logsumexp(w .+ lkl, dims = 1)[:]
end

####
#	Functions for sampling the model
####
Base.rand(m::SumNode) = rand(m.components[sample(Weights(m.prior))])


####
#	Functions for making the library compatible with HierarchicalUtils
####
HierarchicalUtils.nodeshow(io::IO, ::SumNode) = print(io, "SumNode")
HierarchicalUtils.NodeType(::Type{<:SumNode}) = InnerNode()
HierarchicalUtils.printchildren(node::SumNode) = tuple(node.components...)

