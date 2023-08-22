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
using SpecialFunctions

const Maybe{T} = Union{T, Missing}
const MaybeHotArray{T} = Union{MaybeHotVector{T}, MaybeHotMatrix{T}}
const Code = Union{AbstractVector{<:Integer}, Base.CodeUnits}
const Sequence = Union{AbstractString, Code}

include("modelnodes/modelnode.jl")
include("distributions/distributions.jl")
include("modelbuilders.jl")
include("util.jl")
include("loss.jl")
include("reflectinmodel.jl")

export Poisson, Geometric, Categorical, MvNormal, MvNormalParams
export logpdf, logjnt
export SumNode, ProductNode, SetNode
export rand, randwithlabel
export gmm, setmixture, sharedsetmixture, spn

export reflectinmodel
export em_loss, ce_loss

Base.show(io::IO, ::MIME"text/plain", n::AbstractModelNode) = HierarchicalUtils.printtree(io, n)


end # end module
