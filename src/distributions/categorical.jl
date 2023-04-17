
mutable struct _Categorical{T} <: _Distribution{T}
    logp::Array{T, 1}
end
Flux.@functor _Categorical
_Categorical(n::Integer; dtype::Type{<:Real}=Float32) = _Categorical(ones(dtype, n))

function logpdf(m::_Categorical, x::Union{Int, Vector{Int}})
    logp = logsoftmax(m.logp)
    logp[x]
end

logpdf(m::_Categorical, x::Matrix)       = logpdf(m, vec(x))
logpdf(m::_Categorical, x::Real)         = logpdf(m, convert.(Int64, x))
logpdf(m::_Categorical, x::Vector{Real}) = logpdf(m, convert.(Int64, x))
logpdf(m::_Categorical, x::Matrix{Real}) = logpdf(m, convert.(Int64, vec(x)))

logpdf(m::_Categorical, x::Union{OneHotArray, MaybeHotArray}) = vec(_logpdf(m, x))
_logpdf(m::_Categorical, x::OneHotArray) = reshape(logsoftmax(m.logp), 1, :) * x
_logpdf(m::_Categorical{T}, x::MaybeHotArray) where {T<:Real} = PostImputingMatrix(reshape(logsoftmax(m.logp), 1, :)) * x

Base.length(m::_Categorical) = 1
Base.rand(m::_Categorical) = sample(Weights(softmax(m.logp)))
Base.rand(m::_Categorical, ns::Int...) = sample(1:length(m.logp), Weights(softmax(m.logp)), ns)
