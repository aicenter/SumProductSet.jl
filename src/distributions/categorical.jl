
mutable struct _Categorical{T} <: _Distribution{T}
    logp::Array{T, 1}
end

Flux.@functor _Categorical

_Categorical(n::Int; dtype::Type{<:Real}=Float64) = _Categorical(ones(dtype, n))


Base.length(m::_Categorical) = 1
Base.rand(m::_Categorical) = sample(Weights(softmax(m.logp)))
Base.rand(m::_Categorical, ns::Int...) = sample(1:length(m.logp), Weights(softmax(m.logp)), ns)

function logpdf(m::_Categorical, x::Union{Int, Vector{Int}})
    logp = logsoftmax(m.logp)
    logp[x]
end


# only for `x` inputs whose elements can be losslesly converted to integers
# TODO: Add args check
logpdf(m::_Categorical, x::Matrix) = logpdf(m, vec(x)) 
logpdf(m::_Categorical, x::Union{Float64, Vector{Float64}}) = logpdf(m, convert.(Int64, x))
logpdf(m::_Categorical, x::Union{Matrix{Float64}}) = logpdf(m, convert.(Int64, vec(x)))

_oh_logpdf(m::_Categorical, x::Flux.OneHotArray) = reshape(logsoftmax(m.logp), 1, :) * x
logpdf(m::_Categorical, x::Flux.OneHotMatrix) = vec(_oh_logpdf(m, x))
logpdf(m::_Categorical, x::Flux.OneHotVector) = _oh_logpdf(m, x)[]

