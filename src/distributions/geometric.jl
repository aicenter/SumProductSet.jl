"""
    Geometric{T} <: Distribution

Implement multivariate (as well as univariate) geometric distribution as `Distribution`. The distribution is parametrized 
by a vector of real numbers `logitp`, whose elementwise `sigmoid`` represent success probabilities parameters. 
The distribution represents vectorized version of product of independent univariate geometric
distributions.

# Examples
```julia
julia> Random.seed!(0);

julia> m = Geometric(4)
Geometric

julia> x = rand(m, 2)
4×2 Mill.ArrayNode{SparseArrays.SparseMatrixCSC{Int64, Int64}, Nothing}:
 5  1
 ⋅  ⋅
 ⋅  1
 ⋅  ⋅

julia> logpdf(m, x)
1×2 Matrix{Float32}:
 -6.24837  -4.15767
```

"""


struct Geometric{T} <: Distribution
    logitp::Array{T, 1}
end

Flux.@functor Geometric

Geometric(n::Int; dtype::Type{<:Real}=Float32) = Geometric(dtype(0.01)*randn(dtype, n))

####
#   Functions for calculating the likelihood
####

# TODO: precompute logsigmoid.(m.logitp)


function _logpdf_geometric(logitp::Vector{T}, x::SparseMatrixCSC) where {T<:Real}
    linit = sum(logsigmoid, logitp)
    l = fill(linit, 1, size(x, 2))
    @inbounds for (i, j, k) in zip(findnz(x)...)
        l[j] += k*logsigmoid(-logitp[i]) 
    end
    l
end

function _logpdf_back(logitp::Vector{T}, x, Δy) where {T<:Real}
    Δlogitp = fill(sum(Δy), size(x, 1))

    @inbounds for r in eachindex(Δlogitp)
        Δlogitp[r] *= sigmoid(-logitp[r])
    end

    @inbounds for (i, j, k) in zip(findnz(x)...)
        Δlogitp[i] -= k*sigmoid(logitp[i])*Δy[j] 
    end
    Δlogitp, NoTangent()
end

function ChainRulesCore.rrule(::typeof(_logpdf_geometric), logitp::Vector{T}, x::SparseMatrixCSC) where {T<:Real}
    y = _logpdf_geometric(logitp, x)
    _logpdf_pullback = Δy -> (NoTangent(), _logpdf_back(logitp, x, Δy)...)
    return y, _logpdf_pullback
end

logpdf(m::Geometric, x::SparseMatrixCSC) = _logpdf_geometric(m.logitp, x)
logpdf(m::Geometric, x::NGramMatrix) = logpdf(m, SparseMatrixCSC(x))

# _logpdf(m::Geometric, k::SparseMatrixCSC) = k .*logsigmoid.(-m.logitp) .+ logsigmoid.(m.logitp)

# logpdf(m::Geometric,     x::NGramMatrix{T})         where {T<:Sequence}            = sum(_logpdf(m, x); dims=1)
# logpdf(m::Geometric{Tm}, x::NGramMatrix{Maybe{Tx}}) where {Tm<:Real, Tx<:Sequence} = sum(coalesce.(_logpdf(m, x), Tm(0e0)); dims=1)
# logpdf(m::Geometric, x::SparseMatrixCSC) = sum(_logpdf(m, x); dims=1)

####
    #   Functions for generating random samples
####

Base.rand(m::Geometric, n::Int) = ArrayNode(SparseMatrixCSC(floor.(Int, log.(rand(length(m), n)) ./ logsigmoid.(-m.logitp))))
Base.rand(m::Geometric) = rand(m, 1)

####
#   Utilities
####

Base.length(m::Geometric) = length(m.logitp)
