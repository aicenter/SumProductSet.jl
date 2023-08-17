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

function logpdf(m::Categorical, x::Union{Int, Vector{Int}})
    logp = logsoftmax(m.logp)
    logp[x]
end

logpdf(m::Categorical, x::Matrix)         = logpdf(m, vec(x))
logpdf(m::Categorical, x::Real)           = logpdf(m, convert.(Int64, x))
logpdf(m::Categorical, x::Vector{<:Real}) = logpdf(m, convert.(Int64, x))
logpdf(m::Categorical, x::Matrix{<:Real}) = logpdf(m, convert.(Int64, vec(x)))

_logpdf(m::Categorical, x::OneHotArray)   =                    reshape(logsoftmax(m.logp), 1, :)  * x
_logpdf(m::Categorical, x::MaybeHotArray) = PostImputingMatrix(reshape(logsoftmax(m.logp), 1, :)) * x

logpdf(m::Categorical, x::Union{OneHotArray, MaybeHotArray}) = _logpdf(m, x)

####
#   Functions for generating random samples
####

Base.rand(m::Categorical, n::Int) = (r=1:length(m.logp); Flux.onehotbatch(sample(r, Weights(softmax(m.logp)), n), r) |> Mill.ArrayNode)
Base.rand(m::Categorical) = rand(m, 1)

####
#   Utilities
####

Base.length(m::Categorical) = 1
