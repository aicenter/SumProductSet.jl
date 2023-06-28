module SumProductSet

using Flux
using NNlib
using StatsBase
using HierarchicalUtils
using PoissonRandom
using LinearAlgebra
using OneHotArrays
using SparseArrays
using Random
using Mill

const Maybe{T} = Union{T, Missing}
const MaybeHotArray{T} = Union{MaybeHotVector{T}, MaybeHotMatrix{T}}
const Code = Union{AbstractVector{<:Integer}, Base.CodeUnits}
const Sequence = Union{AbstractString, Code}

logfactorial(x::Real) = sum(log.(collect(2:x)))

include("modelnodes/modelnode.jl")
include("distributions/distributions.jl")
include("modelbuilders.jl")
include("util.jl")
include("leaves/vae.jl")
include("reflectinmodel.jl")
include("loss.jl")

export ZIPoisson, Poisson, Geometric, Categorical, MvNormal, MvNormalParams
export logpdf, logjnt
export SumNode, ProductNode, SetNode
export rand, randwithlabel
export setmixture, gmm, sharedsetmixture, spn

export reflectinmodel
export ul_loss, sl_loss, ssl_loss

export VAE, Encoder, Decoder, SplitLayer, elbo, reconstruct_loss

Base.show(io::IO, ::MIME"text/plain", n::AbstractModelNode) = HierarchicalUtils.printtree(io, n)


end # end module
