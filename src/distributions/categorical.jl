
mutable struct _Categorical{T} <: _Distribution{T}
    logp::Array{T, 1}
    # function _Categorical(lp::Array{T, 1}) where {T <: Real}
    #     _Categorical([lp..., 1])
    # end
end

Flux.@functor _Categorical

_Categorical(n::Int; dtype::Type{<:Real}=Float64) = _Categorical(ones(dtype, n))


Base.length(m::_Categorical) = 1
Base.rand(m::_Categorical) =  sample(Weights(softmax(m.logp)))
Base.rand(m::_Categorical, n::Int) =  sample(1:length(m.logp), Weights(softmax(m.logp)), n)

# logpdf(m::_Categorical, x::Int) = x > 0 && x <= length(m.logp)-1 ? m.logp[x] : m.logp[end]
# logpdf(m::_Categorical, x::Vector{Int}) = map(xi -> logpdf(m, xi), x)

function logpdf(m::_Categorical, x::Union{Int, Vector{Int}})
    logp = logsoftmax(m.logp)
    logp[x]
end

# only for `x` inputs whose elements can be losslesly converted to integers
function logpdf(m::_Categorical, x::Union{Float64, Vector{Float64}})
    logp = logsoftmax(m.logp)
    logp[convert.(Int64, x)]
end

# only for matrices of size 1 x n
function logpdf(m::_Categorical, x::Matrix{Float64})
    logp = logsoftmax(m.logp)
    logp[convert.(Int64, vec(x))]
end
