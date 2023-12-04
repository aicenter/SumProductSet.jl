
struct MvStudentt{T}  <: Distribution
    m::Vector{T}
    logs::Vector{T}
    nu::Int
end

Flux.@functor MvStudentt

MvStudentt(d::Int; dtype::Type{<:Real}=Float32) = MvStudentt(randn(dtype, d), rand(dtype, d), 3)


# TODO: Fix mv student problem with dimesion, see wiki

function logpdf(m::MvStudentt{T}, x::Matrix{T}) where T<:Real
    d = size(x, 1)
    z = sum(((x .- m.m) ./ exp.(m.logs)).^2 /m.nu, dims=1)
    -log.(1 .+ z) .*(m.nu+d)/2 .+ loggamma((m.nu+d)/2) .- loggamma(m.nu/2) .- d/2 * log(T(π)* m.nu) .- logsumexp(m.logs)/2
end

Base.length(m::MvStudentt) = length(m.m)

