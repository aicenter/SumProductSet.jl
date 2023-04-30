
mutable struct Poisson{T} <: Distribution{T}
    logλ::Array{T, 1}
    function Poisson(logλ::Array{T, 1}) where T
        @assert length(logλ) == 1
        new{T}(logλ)
    end
end

Flux.@functor Poisson

Poisson(logλ::Real) = Poisson([logλ])
Poisson(logλ::Integer) = Poisson(Float32(logλ))
Poisson() = Poisson(log(rand(2:5)))

####
#   Functions for calculating the likelihood
####

_logpdf(logλ::Real, x) = x .* logλ .- exp(logλ) .- logfactorial.(x)

logpdf(m::Poisson, x::Union{Real, Vector{<:Real}}) = mapreduce(logλ -> _logpdf(logλ, x), +, m.logλ)

####
#   Functions for generating random samples
####

Base.rand(m::Poisson) = pois_rand(exp(m.logλ[1]))
Base.rand(m::Poisson, n::Int) = map(_->rand(m), 1:n)

####
#   Utilities
####

Base.length(m::Poisson) = length(m.logλ)
