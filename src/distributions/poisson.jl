"""
    Poisson{T} <: Distribution

Implement multivariate (as well as univariate) Poisson distribution as `Distribution`. The distribution is parametrized 
by a vector of real numbers `lograte`, whose elementwise `exp` function represents rate parameters. 

# Examples
```julia
julia> Random.seed!(0);

julia> m = Poisson(4)
Poisson

julia> x = rand(m, 2)
4×2 Mill.ArrayNode{Matrix{Int64}, Nothing}:
  7   7
  2   2
 14  10
  0   1

1×2 Matrix{Float64}:
 -8.99603  -7.00497

```

"""

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
_logfactorial(x; dtype::Type{<:Real}=Float64) = sum(log.(2:x))
_logpdf(lograte, x) = x .* lograte .- exp.(lograte) .- _logfactorial.(x)
logpdf(m::Poisson, x::Matrix{<:Real}) = sum(_logpdf(m.lograte, x), dims=1)
logpdf(m::Poisson, x::Vector{<:Real}) = sum(_logpdf(m.lograte, hcat(x...)), dims=1)
logpdf(m::Poisson, x::Real) = hcat(_logpdf(m.lograte, x))  # for consistency

####
#   Functions for generating random samples
####

Base.rand(m::Poisson, n::Int) = Mill.ArrayNode([pois_rand(exp(logr)) for logr in m.lograte, _ in 1:n])
Base.rand(m::Poisson) = rand(m, 1)

####
#   Utilities
####

Base.length(m::Poisson) = length(m.lograte)
