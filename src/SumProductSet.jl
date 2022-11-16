module SumProductSet

using Flux
using NNlib
using StatsBase
using HierarchicalUtils
using PoissonRandom
using LinearAlgebra: det, diagm, cholesky
using Random
import Mill

function logfactorial(x::Real)
    # sum(log.(collect(2:x)); init=zero(Float64))
    sum(log.(collect(2:x)))
end

include("distributions.jl")
include("setnode.jl")
include("sumnode.jl")
include("productnode.jl")
include("modelbuilders.jl")
include("utils.jl")
include("leafs/vae.jl")

export _Poisson, _Categorical, _MvNormal, _MvNormalParams
export logpdf, logjnt
export SumNode, ProductNode, SetNode
export rand, randwithlabel
export setmixture, gmm, sharedsetmixture

export VAE, Encoder, Decoder, SplitLayer, elbo, reconstruct_loss

Base.show(io::IO, ::MIME"text/plain", n::Union{SumNode, SetNode, ProductNode, _Distribution}) = HierarchicalUtils.printtree(io, n)


end # end module