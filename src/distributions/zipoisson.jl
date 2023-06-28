
# Zero Inflated Poisson Distribution

mutable struct ZIPoisson{T} <: Distribution{T}
    logλ::Vector{T}
    logitp::Vector{T}
    function ZIPoisson(logλ::Vector{T}, logitp::Vector{T}) where T
        @assert length(logλ) == length(logitp)
        new{T}(logλ, logitp)
    end
end

Flux.@functor ZIPoisson


ZIPoisson(n::Integer) = ZIPoisson(log.(rand(2:10, n)), zeros(n))

####
#   Functions for calculating the likelihood
####

# note: log p = logsigmoid(w) and log(1-p) = logsigmoid(-w)

_logpdf(m::ZIPoisson, x) = logsigmoid.(-m.logitp) .+ x.*m.logλ .- exp.(m.logλ) .- logfactorial.(x) + (x.==0) .* logsigmoid.(m.logitp)
logpdf(m::ZIPoisson, x) = sum(_logpdf(m, x), dims=1)

logpdf(m::ZIPoisson, x::NGramMatrix{T}) where {T<:Sequence} = logpdf(m, SparseMatrixCSC(x))
logpdf(m::ZIPoisson{Tm}, x::NGramMatrix{Maybe{Tx}}) where {Tm<:Real, Tx<:Sequence} = sum(coalesce.(_logpdf(m, SparseMatrixCSC(x)), Tm(0e0)); dims=1)

####
#   Functions for generating random samples
####

function Base.rand(m::ZIPoisson, n::Int) 
    [rand() < sigmoid(logit) ? 0 : pois_rand(exp(logλ)) for (logλ, logit) in zip(m.logλ, m.logitp), _ in 1:n]
end
Base.rand(m::ZIPoisson) = rand(m, 1)

####
#   Utilities
####

Base.length(m::ZIPoisson) = length(m.logλ)
