"""
    MvNormal{T, N} <: Distribution

Implement Normal distribution as `Distribution`. The distribution is parametrized 
by a vector `b`, `N`-dimensional array `A` and regularization factor `r`. 
Mean vector `μ` and convariance matrix `Σ` can be computed as follows:
- `μ = -A⁻¹ b`,
- `Σ = A⁻ᵀ A⁻¹`.
Regularization factor `r` regularizes `A`, not `Σ`.

# Examples
```julia
julia> Random.seed!(0);

julia> m = MvNormal(3)
MvNormal

julia> x = rand(m, 2)
3×2 Mill.ArrayNode{Matrix{Float32}, Nothing}:
  1.4966074   0.2833557
 -0.0646999  -0.4232424
 -2.1708894  -2.0524693

julia> logpdf(m, x)
1×2 Matrix{Float32}:
 -4.65435  -3.42413
```

"""
struct MvNormal{T, N} <: Distribution
    b::Array{T, 1}
    A::Array{T, N}
    r::T
end

Flux.@functor MvNormal
Flux.trainable(m::MvNormal) = (m.b, m.A,)

MvNormal{T, N}(b::Array{T, 1}, A::Array{T, N}) where {T, N} = MvNormal(b, A, zero(T))

function MvNormal(d::Integer; dtype::Type{<:Real}=Float32, stype::Symbol=:full, minit::Symbol=:uniform, sinit::Symbol=:unit, r::Real=0.)
    m = select_m(minit, dtype, d)
    s = select_s(sinit, dtype, d)
    stype == :full   && return MvNormalParams(m, diagm(s), dtype(r))
    stype == :diag   && return MvNormalParams(m, s,        dtype(r))
    stype == :scalar && return MvNormalParams(m, s[1:1],   dtype(r))
    error("Specified covariance type $(Σtype) is not supported.")
end

####
#   Functions for calculating the likelihood
####

_logpdf(x::Union{Array{T, 2}, Array{Maybe{T}, 2}}) where {T<:Real} = -T(5e-1)*(log(T(2e0)*T(pi)) .+ x.^2)

logpdf(m::MvNormal{T, 2}, x::Array{T, 2})        where {T<:Real} = log(abs(det(m.A + m.r * I)))     .+ sum(_logpdf((m.A  + m.r * I)  * x .+ m.b),          dims=1)
logpdf(m::MvNormal{T, 1}, x::Array{T, 2})        where {T<:Real} = sum(log.(abs.(m.A .+ m.r)))      .+ sum(_logpdf((m.A .+ m.r)     .* x .+ m.b),          dims=1)
logpdf(m::MvNormal{T, 1}, x::Array{Maybe{T}, 2}) where {T<:Real} = sum(coalesce.(log.(abs.(m.A .+ m.r)) .+ _logpdf((m.A .+ m.r)     .* x .+ m.b), T(0e0)); dims=1)
logpdf(m::MvNormal{T, 1}, x::Array{Missing, 2})  where {T<:Real} = zeros(T, 1, size(x, 2))

logpdf(m::MvNormal{T, 2}, x::Array{T, 1})        where {T<:Real} = logpdf(m, hcat(x))
logpdf(m::MvNormal{T, 1}, x::Array{T, 1})        where {T<:Real} = logpdf(m, hcat(x))
logpdf(m::MvNormal{T, 1}, x::Array{Maybe{T}, 1}) where {T<:Real} = logpdf(m, hcat(x))
logpdf(m::MvNormal{T, 1}, x::Array{Missing, 1})  where {T<:Real} = zeros(T, 1, 1)

####
#   Functions for generating random samples
####

Base.rand(m::MvNormal{T, 2}, n::Int) where {T<:Real} =   inv(m.A)  * (randn(T, length(m.b), n) .- m.b) |> Mill.ArrayNode
Base.rand(m::MvNormal{T, 1}, n::Int) where {T<:Real} = (1 ./ m.A) .* (randn(T, length(m.b), n) .- m.b) |> Mill.ArrayNode
Base.rand(m::MvNormal) = rand(m, 1)

####
#   Utilities
####

Base.length(m::MvNormal) = length(m.b)

function select_s(selector, dtype, d)
    selector == :unit     && return ones(dtype, d)
    selector == :randlow  && return dtype(5e-1) .+ dtype(5e-1)*rand(dtype, d)
    selector == :randhigh && return dtype(1e-0) .+ dtype(5e-1)*rand(dtype, d)
    error("Specified covariance initialization $(selector) is not supported.")
end
function select_m(selector, dtype, d)
    selector == :uniform  && return dtype(2e0)*rand(dtype, d) .- dtype(1e0)
    selector == :randn  && return randn(dtype, d)
    selector == :zero && return zeros(dtype, d)
    error("Specified mean initialization $(selector) is not supported.")
end

function MvNormalParams(μ::Array{T, 1}, Σ::Array{T, 2}, r::T=zero(T)) where {T<:Real}
    A = Matrix(cholesky(inv(Σ)).U)
    b = -A * μ
    MvNormal{T, 2}(b, A, r)
end
function MvNormalParams(μ::Array{T, 1}, Σ::Array{T, 1}, r::T=zero(T)) where {T<:Real}
    A = 1 ./ sqrt.(Σ)
    b = - A .* μ
    MvNormal{T, 1}(b, A, r)
end
