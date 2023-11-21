module SumProductSet

using Flux
using ChainRulesCore
using NNlib
using StatsBase
using HierarchicalUtils
using PoissonRandom
using LinearAlgebra
using OneHotArrays
using SparseArrays
using Random
using Mill
using SpecialFunctions

const Maybe{T} = Union{T, Missing}
const MaybeHotArray{T} = Union{MaybeHotVector{T}, MaybeHotMatrix{T}}
const Code = Union{AbstractVector{<:Integer}, Base.CodeUnits}
const Sequence = Union{AbstractString, Code}

include("modelnodes/modelnode.jl")
include("distributions/distributions.jl")
include("modelbuilders.jl")
include("util.jl")
include("reflectinmodel.jl")
include("loss.jl")
include("rank.jl")

export Poisson, Geometric, Categorical, MvNormal, MvNormalParams
export DiagRMvNormal, IsoRMvNormal, UnitMvNormal
export logpdf, logjnt
export SumNode, ProductNode, SetNode
export rand, randwithlabel
export setmixture, gmm, sharedsetmixture, spn

export reflectinmodel
export em_loss, ce_loss, ssl_loss
export rank

export VAE, Encoder, Decoder, SplitLayer, elbo, reconstruct_loss

Base.show(io::IO, ::MIME"text/plain", n::AbstractModelNode) = HierarchicalUtils.printtree(io, n)
# Base.show(io::IO, n::AbstractModelNode) = print(io, "Node")
# Base.show(io::IO, n::AbstractArray{<:AbstractModelNode}) = print(io, "Node")

end # end module
