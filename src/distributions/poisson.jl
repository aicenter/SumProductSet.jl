
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
