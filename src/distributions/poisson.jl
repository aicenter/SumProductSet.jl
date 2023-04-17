
mutable struct _Poisson{T} <: _Distribution{T}
    logλ::Array{T, 1}
    function _Poisson(logλ::Array{T, 1}) where T
        @assert length(logλ) == 1
        new{T}(logλ)
    end
end
Flux.@functor _Poisson

_Poisson(logλ::Real) = _Poisson([logλ])
_Poisson(logλ::Integer) = _Poisson(Float32(logλ))
_Poisson() = _Poisson(log(rand(2:5)))

Base.rand(m::_Poisson) = pois_rand(exp(m.logλ[1]))
Base.rand(m::_Poisson, n::Int) = map(_->rand(m), 1:n)
Base.length(m::_Poisson) = length(m.logλ)

_poisson_logpdf(logλ::Real, x) = x .* logλ .- exp(logλ) .- logfactorial.(x)

function logpdf(m::_Poisson, x::Union{Real, Vector{<:Real}})
    mapreduce(logλ -> _poisson_logpdf(logλ, x), +, m.logλ)
end
