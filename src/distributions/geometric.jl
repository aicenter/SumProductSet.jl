
mutable struct _Geometric{T} <: _Distribution{T}
    logp::Array{T, 1}
end
Flux.@functor _Geometric
_Geometric(n::Int; dtype::Type{<:Real}=Float32) = _Geometric(ones(dtype, n))

_logpdf(m::_Geometric, k) = SparseMatrixCSC(k).*log1p.(m.logp) .+ m.logp

logpdf(m::_Geometric, x::NGramMatrix{T}) where {T<:Sequence} = vec(sum(_logpdf(m, x); dims=1))
logpdf(m::_Geometric{Tm}, x::NGramMatrix{Maybe{Tx}}) where {Tm<:Real, Tx<:Sequence} = vec(sum(coalesce.(_logpdf(m, x), Tm(0e0)); dims=1))

Base.length(m::_Geometric) = 1
