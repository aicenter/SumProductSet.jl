
mutable struct _MvNormal{T} <: ContinuousMultivariateDistribution
    b::Array{T,1}
    A::Array{T,2}
end
Flux.@functor _MvNormal

function _MvNormalParams(μ::Array{T, 1}, Σ::Array{T, 2}) where {T<:Real}
    A = Matrix(cholesky(inv(Σ)).U)
    b = - A * μ
    _MvNormal(b, A)
end

_MvNormal(d::Int) =_MvNormal(zeros(Float64, d), diagm(ones(Float64, d)))

function Base.rand(m::_MvNormal{T}, n::Int) where {T<:Real} 
    inv(m.A) * (randn(T, length(m.b), n) .- m.b)
end

Base.rand(m::_MvNormal) = vec(rand(m, 1))

Base.length(m::_MvNormal) = length(m.b)

function Distributions.logpdf(m::_MvNormal{T}, x::Array{T, 2}) where {T<:Real}
    z = m.A * x .+ m.b
    l = log(abs(det(m.A))) .- T(5e-1)*(size(z, 1)*log(T(2e0)*T(pi)) .+ sum(z.^2, dims=1))
    l[:]
end

Distributions.logpdf(m::_MvNormal{T}, x::Array{T, 1}) where {T<:Real} = logpdf(m, hcat(x))
