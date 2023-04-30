
mutable struct Categorical{T} <: Distribution{T}
    logp::Array{T, 1}
end
Flux.@functor Categorical
Categorical(n::Integer; dtype::Type{<:Real}=Float32) = Categorical(ones(dtype, n))

function logpdf(m::Categorical, x::Union{Int, Vector{Int}})
    logp = logsoftmax(m.logp)
    logp[x]
end

logpdf(m::Categorical, x::Matrix)         = logpdf(m, vec(x))
logpdf(m::Categorical, x::Real)           = logpdf(m, convert.(Int64, x))
logpdf(m::Categorical, x::Vector{<:Real}) = logpdf(m, convert.(Int64, x))
logpdf(m::Categorical, x::Matrix{<:Real}) = logpdf(m, convert.(Int64, vec(x)))

logpdf(m::Categorical, x::Union{OneHotArray, MaybeHotArray}) = vec(_logpdf(m, x))
_logpdf(m::Categorical, x::OneHotArray) = reshape(logsoftmax(m.logp), 1, :) * x
_logpdf(m::Categorical, x::MaybeHotArray) = PostImputingMatrix(reshape(logsoftmax(m.logp), 1, :)) * x

Base.length(m::Categorical) = 1
Base.rand(m::Categorical) = sample(Weights(softmax(m.logp)))
Base.rand(m::Categorical, ns::Int...) = sample(1:length(m.logp), Weights(softmax(m.logp)), ns)
