
struct MvBernoulli{T} <: Distribution
    logitp::Vector{T}
end

Flux.@functor MvBernoulli

MvBernoulli(n::Int; dtype::Type{<:Real}=Float32) = MvBernoulli(dtype(0.1)*randn(dtype, n))

####
#   Functions for calculating the likelihood
###
logpdf(m::MvBernoulli, x::BitMatrix) = _logpdf_bern(m.logitp, x)


function _logpdf_bern(logitp::Vector{T}, x::BitMatrix) where T
    lp = logsigmoid.(logitp)
    l1p = logsigmoid.(-logitp)
    l = zeros(T, 1, size(x, 2))
    @inbounds for j in 1:size(x, 2)
        for i in 1:size(x, 1)
            if x[i, j]
                l[j] += lp[i]
            else
                l[j] += l1p[i]
            end
        end
    end
    l
end


function _logpdf_bern_back(logitp::Vector{T}, x::BitMatrix, Δy) where T
    p = sigmoid.(logitp)
    Δp = zero(logitp)
    @inbounds for j in 1:size(x, 2)
        for i in 1:size(x, 1)
            if x[i, j]
                Δp[i] += (1-p[i]) * Δy[j]
            else
                Δp[i] -= p[i] * Δy[j]
            end
        end
    end

    Δp, NoTangent()
end


function ChainRulesCore.rrule(::typeof(_logpdf_bern), logitp::Vector{T}, x::BitMatrix) where {T<:Real}
    y = _logpdf_bern(logitp, x)
    _logpdf_pullback = Δy -> (NoTangent(), _logpdf_bern_back(logitp, x, Δy)...)
    return y, _logpdf_pullback
end

    
####
    #   Functions for generating random samples
####

Base.rand(m::MvBernoulli, n::Int) = ArrayNode(BitMatrix(rand(length(m), n) .< sigmoid.(m.logitp)))
Base.rand(m::MvBernoulli) = rand(m, 1)

####
#   Utilities
####

Base.length(m::MvBernoulli) = length(m.logitp)
