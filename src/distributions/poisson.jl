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
Poisson(n::Int; dtype::Type{<:Real}=Float32) = Poisson(dtype.(log.(rand(2:10, n))))
Poisson(dtype::Type{<:Real}=Float32) = Poisson(1; dtype=dtype)

####
#   Functions for calculating the likelihood
####

function _logpdf_poisson(lograte::Vector{T}, x::Matrix{<:Real}) where {T<:Real}
    ndims, nobs = size(x)
    linit = -sum(exp, lograte)

    l = fill(linit, 1, nobs)
    @inbounds for j in 1:nobs
        for i in 1:ndims
            l[j] += x[i, j] * lograte[i] - logfactorial(x[i, j])
        end
    end
    l
end


function _logpdf_poisson(lograte::Vector{T}, x::SparseMatrixCSC) where {T<:Real}
    linit = -sum(exp, lograte)
    l = fill(linit, 1, size(x, 2))

    rows = rowvals(x)
    vals = nonzeros(x)
    @inbounds for j in 1:size(x, 2)
        for i in nzrange(x, j)
            row = rows[i]
            val = vals[i]
            l[j] += val * lograte[row] - logfactorial(val)
        end
    end
    l
end

function _logpdf_poisson_back(lograte::Vector{T}, x::Matrix{<:Real}, Δy) where {T<:Real}
    ndims, nobs = size(x)
    sum_Δy = sum(Δy)
    Δp = zeros(T, ndims)
    
    @inbounds for i in 1:ndims
        for j in 1:nobs
            Δp[i] += x[i, j] * Δy[j]
        end
        Δp[i] -= sum_Δy * exp(lograte[i])
    end
    Δp, NoTangent()
end

function _logpdf_poisson_back(lograte::Vector{T}, x::SparseMatrixCSC, Δy) where {T<:Real}
    ndims, nobs = size(x)
    sum_Δy = sum(Δy)
    Δp = zeros(T, ndims)

    @inbounds for i in eachindex(lograte)
        Δp[i] -= sum_Δy * exp(lograte[i])
    end

    rows = rowvals(x)
    vals = nonzeros(x)
    @inbounds for j in 1:size(x, 2)
        for i in nzrange(x, j)
            row = rows[i]
            val = vals[i]
            Δp[row] += val * Δy[j]
        end
    end
    
    Δp, NoTangent()
end


function ChainRulesCore.rrule(::typeof(_logpdf_poisson), args...)
    _logpdf_pullback = Δy -> (NoTangent(), _logpdf_poisson_back(args..., Δy)...)
    _logpdf_poisson(args...), _logpdf_pullback
end

logpdf(m::Poisson, x::Matrix{<:Real}) = _logpdf_poisson(m.lograte, x)
logpdf(m::Poisson, x::SparseMatrixCSC) = _logpdf_poisson(m.lograte, x)
logpdf(m::Poisson, x::Union{T, Vector{T}} where T<:Real) = logpdf(m, hcat(x...))

# old logpdfs
# _logpdf(lograte, x) = x .* lograte .- exp.(lograte) .- _logfactorial.(x)
# logpdf(m::Poisson, x::Matrix{<:Real}) = sum(_logpdf(m.lograte, x), dims=1)
# logpdf(m::Poisson, x::Vector{<:Real}) = sum(_logpdf(m.lograte, hcat(x...)), dims=1)
# logpdf(m::Poisson, x::Real) = hcat(_logpdf(m.lograte, x))  # for consistency
# logpdf(m::Poisson, x::SparseMatrixCSC) = sum(_logpdf(m.lograte, x), dims=1)

####
#   Functions for generating random samples
####

# pois_rand does not work with Float32
Base.rand(m::Poisson, n::Int) = Mill.ArrayNode([pois_rand(exp(Float64.(logr))) for logr in m.lograte, _ in 1:n])
Base.rand(m::Poisson) = rand(m, 1)

####
#   Utilities
####

Base.length(m::Poisson) = length(m.lograte)
