
# Zero Inflated Poisson Distribution

mutable struct ZIPoisson{T} <: Distribution
    lograte::Vector{T}
    logitp::Vector{T}
    function ZIPoisson(lograte::Vector{T}, logitp::Vector{T}) where T
        @assert length(lograte) == length(logitp)
        new{T}(lograte, logitp)
    end
end

Flux.@functor ZIPoisson


ZIPoisson(n::Integer) = ZIPoisson(log.(rand(2:10, n)), zeros(n))

####
#   Functions for calculating the likelihood
####

# note: log p = logsigmoid(w) and log(1-p) = logsigmoid(-w)

_logfactorial(x; dtype::Type{<:Real}=Float64) = sum(log.(2:x))

# ineffective
function _logpdf(m::ZIPoisson, x)
    log_poisson = x.*m.lograte .- exp.(m.lograte) .- _logfactorial.(x)
    (x.>0).*(logsigmoid.(-m.logitp) .+ log_poisson) + (x.==0).*log.(sigmoid.(m.logitp) .+ sigmoid.(-m.logitp) .* exp.(log_poisson))
end

logpdf(m::ZIPoisson, x::SparseMatrixCSC) = sum(_logpdf(m, x), dims=1)
logpdf(m::ZIPoisson, x::NGramMatrix{T}) where {T<:Sequence} = sum(_logpdf(m, SparseMatrixCSC(x)), dims=1)
logpdf(m::ZIPoisson{Tm}, x::NGramMatrix{Maybe{Tx}}) where {Tm<:Real, Tx<:Sequence} = sum(coalesce.(_logpdf(m, SparseMatrixCSC(x)), Tm(0e0)); dims=1)

####
#   Functions for generating random samples
####

# ineffective
function Base.rand(m::ZIPoisson, n::Int) 
    x = [rand() < sigmoid(logit) ? 0 : pois_rand(exp(logr)) for (logr, logit) in zip(m.lograte, m.logitp), _ in 1:n]
    SparseMatrixCSC(x)
end
Base.rand(m::ZIPoisson) = rand(m, 1)

####
#   Utilities
####

Base.length(m::ZIPoisson) = length(m.lograte)
