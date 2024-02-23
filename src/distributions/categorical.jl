"""
    Categorical{T} <: Distribution

Implement univariate categorical distribution as `Distribution`. The distribution is parametrized 
by a vector of real numbers `logp`, whose `softmax`` represent event/category probabilities parameters.

# Examples
```julia
julia> Random.seed!(0);

julia> m = Categorical(4)
Categorical

julia> x = rand(m, 2)
4×2 Mill.ArrayNode{OneHotArrays.OneHotMatrix{UInt32, 4, Vector{UInt32}}, Nothing}:
 ⋅  1
 1  ⋅
 ⋅  ⋅
 ⋅  ⋅

julia> logpdf(m, x)
1×2 Matrix{Float32}:
 -1.38629  -1.38629
```

"""
struct Categorical{T} <: Distribution
    logp::Vector{T}
end

Flux.@functor Categorical

Categorical(n::Integer; dtype::Type{<:Real}=Float32) = Categorical(ones(dtype, n))

####
#   Functions for calculating the likelihood
####

logpdf(m::Categorical, x::Union{OneHotArray, MaybeHotArray}) = _logpdf_cat(m.logp, x)
logpdf(m::Categorical{T}, x::MaybeHotMatrix{Missing}) where {T<:Real} = zeros(T, 1, size(x, 2))

_get_inidices(x::OneHotArray) = x.indices
_get_inidices(x::MaybeHotMatrix) = x.I
_get_inidices(x::MaybeHotVector) = x.i

l_cat!(_ , _, _, idx::Missing) = nothing
l_cat!(l, logp, i, idx) = l[i] += logp[idx]

function _logpdf_cat(logp::Vector{T}, x) where T
    logp = logsoftmax(logp)
    l = zeros(T, 1, size(x, 2))

    @inbounds for (j, idx) in enumerate(_get_inidices(x))
        l_cat!(l, logp, j, idx)
    end
    l
end

Δlogp_cat!(_, _, idx::Missing, _) = nothing
Δlogp_cat!(Δlogp, Δy, idx, j) = Δlogp[idx] += Δy[j]

_weight(i::Missing, _, ::Type{T}) where T = zero(T)
_weight(_, _, ::Type{T}) where T = one(T)

function _logpdf_cat_back(logp::Vector{T}, x, Δy) where {T <: Real}
    sum_Δy = zero(T)
    Δlogp = zero(logp)

    @inbounds for (j, idx) in enumerate(_get_inidices(x))
        Δlogp_cat!(Δlogp, Δy, idx, j)
        sum_Δy += _weight(idx, j, T) * Δy[j]
    end
    p = softmax(logp)
    @inbounds for i in eachindex(logp)
        Δlogp[i] -= sum_Δy * p[i]
    end

    Δlogp, NoTangent()
end

function ChainRulesCore.rrule(::typeof(_logpdf_cat), args...)
    _logpdf_cat_pullback = Δy -> (NoTangent(), _logpdf_cat_back(args..., Δy)...)
    _logpdf_cat(args...), _logpdf_cat_pullback
end

####
#   Functions for generating random samples
####

Base.rand(m::Categorical, n::Int) = (r=1:length(m.logp); Flux.onehotbatch(sample(r, Weights(softmax(m.logp)), n), r) |> Mill.ArrayNode)
Base.rand(m::Categorical) = rand(m, 1)

####
#   Utilities
####

Base.length(m::Categorical) = 1
