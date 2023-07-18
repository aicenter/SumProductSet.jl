
struct Categorical{T} <: Distribution
    logp::Array{T, 1}
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
