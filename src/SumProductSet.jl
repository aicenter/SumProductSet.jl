module SumProductSet

using Flux
using Distributions
using StatsBase
using HierarchicalUtils
using PoissonRandom
using LinearAlgebra: logdet, diagm


import Mill

function logsumexp(x; dims = :)
	xm = maximum(x, dims = dims)
	log.(sum(exp.(x .- xm), dims = dims)) .+ xm
end

function logfactorial(x::Real)
    # sum(log.(collect(2:x)); init=zero(Float64))
	sum(log.(collect(2:x)))
end

include("distributions.jl")
include("sumnode.jl")
include("productnode.jl")
include("setnode.jl")

export _Poisson, _MvNormal
export logpdf
export SumNode, ProductNode, SetNode

Base.show(io::IO, ::MIME"text/plain", n::Union{SumNode, SetNode, ProductNode, Distribution}) = HierarchicalUtils.printtree(io, n)


end # end module