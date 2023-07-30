
struct Poisson{T <: AbstractFloat} <: Distribution
    lograte::Vector{T}
end

Flux.@functor Poisson

Poisson(lograte::AbstractFloat) = Poisson([lograte])
Poisson(n::Int) = Poisson(Float64.(log.(rand(2:10, n)))) # pois_rand does not work with Float64
Poisson() = Poisson(1)

####
#   Functions for calculating the likelihood
####
_logpdf(lograte, x) = x .* lograte .- exp.(lograte) .- _logfactorial.(x)
logpdf(m::Poisson, x::Matrix{<:Real}) = sum(_logpdf(m.lograte, x), dims=1)
logpdf(m::Poisson, x::Vector{<:Real}) = sum(_logpdf(m.lograte, hcat(x...)), dims=1)
logpdf(m::Poisson, x::Real) = hcat(_logpdf(m.lograte, x))  # for consistency
logpdf(m::Poisson, x::SparseMatrixCSC) = sum(_logpdf(m.lograte, x), dims=1)

####
#   Functions for generating random samples
####

# Base.rand(m::Poisson, n::Int) = mapreduce(logλ -> map(_->pois_rand(exp(logλ)), 1:n)', vcat, m.logλ)
Base.rand(m::Poisson, n::Int) = Mill.ArrayNode([pois_rand(exp(logr)) for logr in m.lograte, _ in 1:n])
Base.rand(m::Poisson) = rand(m, 1)

####
#   Utilities
####

Base.length(m::Poisson) = length(m.lograte)
