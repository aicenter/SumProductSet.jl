
mutable struct _MvNormal{T, N} <: _Distribution{T}
    b::Array{T, 1}
    A::Array{T, N}
    r::T
end
_MvNormal{T, N}(b::Array{T, 1}, A::Array{T, N}) where {T, N} = _MvNormal(b, A, zero(T))

Flux.@functor _MvNormal
Flux.trainable(m::_MvNormal) = (m.b, m.A,)

function _MvNormalParams(μ::Array{T, 1}, Σ::Array{T, 2}, r::T=zero(T)) where {T<:Real}
    A = Matrix(cholesky(inv(Σ)).U)
    b = - A * μ
    _MvNormal{T, 2}(b, A, r)
end

function _MvNormalParams(μ::Array{T, 1}, Σ::Array{T, 1}, r::T=zero(T)) where {T<:Real}
    A = 1 ./ sqrt.(Σ)
    b = - A .* μ
    _MvNormal{T, 1}(b, A, r)
end

function _MvNormal(d::Integer; dtype::Type{<:Real}=Float64, μinit::Symbol=:uniform, Σinit::Symbol=:unit, Σtype::Symbol=:full, r::Real=0.)
    # covariance initialization type selection
    if Σinit == :unit
        diagΣ = ones(dtype, d)
    elseif Σinit == :randlow 
        diagΣ = dtype(0.5) .+ dtype(0.5)*rand(dtype, d)
    elseif Σinit == :randhigh
        diagΣ = dtype(1) .+ dtype(0.5)*rand(dtype, d)
    else
        @error "Specified covariance initialization $(covinit) is not supported."
    end

    # mean initialization type selection
    if μinit == :uniform
        μ = dtype(2)*rand(dtype, d) .- dtype(1)
    elseif μinit == :randn 
        μ = randn(dtype, d)
    elseif μinit == :zero
        μ = zeros(dtype, d)
    else
        @error "Specified mean initialization $(μinit) is not supported."
    end

    # covariance type selection
    if Σtype == :full
        return _MvNormalParams(μ, diagm(diagΣ), dtype(r))
    elseif Σtype == :diag
        return _MvNormalParams(μ, diagΣ, dtype(r))
    elseif Σtype == :scalar
        return _MvNormalParams(μ, diagΣ[1:1], dtype(r))
    else
        @error "Specified covariance type $(Σtype) is not supported."
    end
end

function Base.rand(m::_MvNormal{T, 2}, n::Int) where {T<:Real} 
    inv(m.A) * (randn(T, length(m.b), n) .- m.b)
end
function Base.rand(m::_MvNormal{T, 1}, n::Int) where {T<:Real} 
    (1 ./ m.A) .* (randn(T, length(m.b), n) .- m.b)
end
Base.rand(m::_MvNormal) = rand(m, 1)

Base.length(m::_MvNormal) = length(m.b)

function logpdf(m::_MvNormal{T, 2}, x::Array{T, 2}) where {T<:Real}
    z = (m.A + m.r * I) * x .+ m.b
    l = log(abs(det(m.A + m.r * I))) .- T(5e-1)*(size(z, 1)*log(T(2e0)*T(pi)) .+ sum(z.^2, dims=1))
    l[:]
end
logpdf(m::_MvNormal{T, 2}, x::Array{T, 1}) where {T<:Real} = logpdf(m, hcat(x))

function logpdf(m::_MvNormal{T, 1}, x::Array{T, 2}) where {T<:Real}
    z = (m.A .+ m.r) .* x .+ m.b
    l = sum(log.(abs.(m.A .+ m.r))) .- T(5e-1)*(size(z, 1)*log(T(2e0)*T(pi)) .+ sum(z.^2, dims=1))
    l[:]
end
logpdf(m::_MvNormal{T, 1}, x::Array{T, 1}) where {T<:Real} = logpdf(m, hcat(x))

function logpdf(m::_MvNormal{T, 1}, x::Array{Maybe{T}, 2}) where {T<:Real}
    z = (m.A .+ m.r) .* x .+ m.b
    l = log.(abs.(m.A .+ m.r)) .- T(5e-1)*(log(T(2e0)*T(pi)) .+ z.^2)
    l = coalesce.(l, T(0e0))
    l = sum(l; dims=1)
    l[:]
end
logpdf(m::_MvNormal{T, 1}, x::Array{Maybe{T}, 1}) where {T<:Real} = logpdf(m, hcat(x))
