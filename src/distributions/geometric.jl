
mutable struct Geometric{T} <: Distribution
    logitp::Array{T, 1}
end

Flux.@functor Geometric

Geometric(n::Int; dtype::Type{<:Real}=Float32) = Geometric(dtype(0.01)*randn(dtype, n))

####
#   Functions for calculating the likelihood
####

_logpdf(m::Geometric, k) = _logpdf(m, SparseMatrixCSC(k))
_logpdf(m::Geometric, k::SparseMatrixCSC) = k .*logsigmoid.(-m.logitp) .+ logsigmoid.(m.logitp)

logpdf(m::Geometric,     x::NGramMatrix{T})         where {T<:Sequence}            = mean(_logpdf(m, x); dims=1)
logpdf(m::Geometric{Tm}, x::NGramMatrix{Maybe{Tx}}) where {Tm<:Real, Tx<:Sequence} = mean(coalesce.(_logpdf(m, x), Tm(0e0)); dims=1)
logpdf(m::Geometric, x::SparseMatrixCSC)                                           = mean(_logpdf(m, x); dims=1)

####
    #   Functions for generating random samples
####

Base.rand(m::Geometric, n::Int) = ArrayNode(SparseMatrixCSC(floor.(Int, log.(rand(length(m), n)) ./ logsigmoid.(-m.logitp))))
Base.rand(m::Geometric) = rand(m, 1)

####
#   Utilities
####

Base.length(m::Geometric) = length(m.logitp)
