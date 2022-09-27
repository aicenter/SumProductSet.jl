# Poisson distribution
Distributions.logpdf(d::Distribution, x::Mill.ArrayNode) = logpdf(d, x.data)

mutable struct _Poisson{T} <: DiscreteUnivariateDistribution
    logλ::Array{T, 1}
end
Flux.@functor _Poisson

_Poisson(logλ::Real) = _Poisson([logλ])
_Poisson(logλ::Integer) = _Poisson(float(logλ))

Base.rand(m::_Poisson) = length(m.logλ) > 1 ? throw(error("rand(m::_Poisson) is not defined for length(m.logλ) > 1")) : pois_rand(exp(m.logλ[1]))
Base.length(m::_Poisson) = length(m.logλ)


_poisson_logpdf(logλ, x) = x .* logλ .- exp(logλ) .- logfactorial.(x)

function Distributions.logpdf(m::_Poisson, x::Real)
    mapreduce(logλ -> _poisson_logpdf(logλ, x), +, m.logλ)
end
 
# Normal distribution
mutable struct _MvNormal{T} <: ContinuousMultivariateDistribution
    b::Array{T,1}
    A::Array{T,2}
end
Flux.@functor _MvNormal

_MvNormal(d::Int) =_MvNormal(zeros(Float64, d), diagm(ones(Float64, d)))

Base.rand(m::_MvNormal{T}, n::Int) where {T<:Real} = m.b .+ m.A * randn(T, length(m.b), n)
Base.rand(m::_MvNormal) = vec(rand(m, 1))

Base.length(m::_MvNormal) = length(m.b)

function Distributions.logpdf(m::_MvNormal{T}, x::Array{T, 1}) where {T<:Real}
    z = m.A * x .+ m.b
    logdet(m.A) .- T(5e-1)*(size(z, 1)*log(T(2e0)*T(pi)) .+ sum(z.^2, dims=1))
end

Distributions.logpdf(m::_MvNormal{T}, x::Array{T, 2}) where {T<:Real} = mapreduce(i-> logpdf(m, x[:, i]), vcat, 1:size(x, 2))
####
#  compatibility with HierarchicalUtils
####
HierarchicalUtils.nodeshow(io::IO, ::T) where {T<:Distribution} = print(io, "$(T)")
HierarchicalUtils.NodeType(::Type{<:Distribution}) = LeafNode()
