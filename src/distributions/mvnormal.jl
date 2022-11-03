
mutable struct _MvNormal{T, N} <: _Distribution{T}
    b::Array{T, 1}
    A::Array{T, N}
end
Flux.@functor _MvNormal

function _MvNormalParams(μ::Array{T, 1}, Σ::Array{T, 2}) where {T<:Real}
    A = Matrix(cholesky(inv(Σ)).U)
    b = - A * μ
    _MvNormal{T, 2}(b, A)
end

function _MvNormalParams(μ::Array{T, 1}, Σ::Array{T, 1}) where {T<:Real}
    A = 1 ./ sqrt.(Σ)
    b = - A .* μ
    _MvNormal{T, 1}(b, A)
end

# _MvNormal(d::Int) =_MvNormalParams(randn(Float64, d), diagm(0.5 .+ 0.5*rand(Float64, d)))
# _MvNormal(d::Int) =_MvNormalParams(-2*(-0.5 .+ rand(Float64, d)), diagm(fill(1e0, d)))

function _MvNormal(d::Int; dtype::Type{<:Real}=Float64, μinit::Symbol=:uniform, Σinit::Symbol=:unit, Σtype::Symbol=:full)
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
    else
        @error "Specified mean initialization $(μinit) is not supported."
    end

    # covariance type selection
    if Σtype == :full
        return _MvNormalParams(μ, diagm(diagΣ))
    elseif Σtype == :diag
        return _MvNormalParams(μ, diagΣ)
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
Base.rand(m::_MvNormal) = vec(rand(m, 1))

Base.length(m::_MvNormal) = length(m.b)

function logpdf(m::_MvNormal{T, 2}, x::Array{T, 2}) where {T<:Real}
    z = m.A * x .+ m.b
    l = log(abs(det(m.A))) .- T(5e-1)*(size(z, 1)*log(T(2e0)*T(pi)) .+ sum(z.^2, dims=1))
    l[:]
end
logpdf(m::_MvNormal{T, 2}, x::Array{T, 1}) where {T<:Real} = logpdf(m, hcat(x))

function logpdf(m::_MvNormal{T, 1}, x::Array{T, 2}) where {T<:Real}
    z = m.A .* x .+ m.b
    l = sum(log.(abs.(m.A))) .- T(5e-1)*(size(z, 1)*log(T(2e0)*T(pi)) .+ sum(z.^2, dims=1))
    l[:]
end
logpdf(m::_MvNormal{T, 1}, x::Array{T, 1}) where {T<:Real} = logpdf(m, hcat(x))
