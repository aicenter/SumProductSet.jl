
struct DiagRMvNormal{T} <: Distribution
    m::Array{T, 1}
    logs::Array{T, 1}
    min_s::T
    max_s::T
end

Flux.@functor DiagRMvNormal
DiagRMvNormal(m::Vector{T}, s::Vector{T}) where T = DiagRMvNormal(m, s, T(0.1), T(1.0))

function DiagRMvNormal(d::Integer; min_s=0.01, max_s=1.0, dtype::Type{<:Real}=Float32, minit::Symbol=:uniform, sinit::Symbol=:halfunit)
    m = select_m(minit, dtype, d)
    logs = select_rat_logs(sinit, dtype, d)
    return DiagRMvNormal(m, logs, dtype(min_s), dtype(max_s))
end

####
#   Functions for calculating the likelihood
####
logpdf(m::DiagRMvNormal{T}, x::Matrix{T}) where {T<:Real} = _logpdf_grdiag(m.m, σs(m.logs, m.min_s, m.max_s), x)
σs(logs, min_s, max_s) = min_s .+ (max_s - min_s) .* Flux.sigmoid.(logs)
function _logpdf_grdiag(m::Vector{T}, s::Vector{T}, x::Matrix{T}) where {T<:Real}
    linit = -sum(log, s) - T(5e-1)*size(x, 1) * log(T(2e0)*T(pi))

    l = fill(linit, 1, size(x, 2))
    for j in 1:size(x, 2)
        for i in 1:size(x, 1)
            @inbounds l[j] -=  T(5e-1) .* ((x[i, j] - m[i]) / s[i] ) ^2
        end
    end
    l
end
function logpdf_grdiag_back(m::Vector{T}, s::Vector{T}, x::Matrix{T}, Δy) where {T<:Real}
    Δm = zero(m)
    Δs = -sum(Δy) ./ s

    for j in 1:size(x, 2)
        for i in 1:size(x, 1)
            tmp = (x[i, j] - m[i]) / s[i]
            Δm[i] += Δy[j] * tmp / s[i]
            Δs[i] += Δy[j] * tmp^2 / s[i]
        end
    end
    Δm, Δs, NoTangent()
end
function ChainRulesCore.rrule(::typeof(_logpdf_grdiag), m::Vector{T}, s::Vector{T}, x::Matrix{T}) where {T<:Real}
    y = _logpdf_grdiag(m, s, x)
    _logpdf_pullback = Δy -> (NoTangent(), logpdf_grdiag_back(m, s, x, Δy)...)
    return y, _logpdf_pullback
end

####
#   Functions for generating random samples
####

Base.rand(m::DiagRMvNormal{T}, n::Int) where {T<:Real} =  (σs(m.logs, m.min_s, m.max_s) .* randn(T, length(m.m), n) .+ m.m) |> Mill.ArrayNode
Base.rand(m::DiagRMvNormal) = rand(m, 1)

####
#   Utilities
####

Base.length(m::DiagRMvNormal) = length(m.m)

function select_rat_logs(selector, dtype, d)
    selector == :halfunit     && return zeros(dtype, d)
    # selector == :randlow  && return dtype(5e-1) .+ dtype(5e-1)*rand(dtype, d)
    # selector == :randhigh && return dtype(1e-0) .+ dtype(5e-1)*rand(dtype, d)
    error("Specified covariance initialization $(selector) is not supported.")
end
function select_rat_m(selector, dtype, d)
    selector == :uniform  && return dtype(2e0)*rand(dtype, d) .- dtype(1e0)
    selector == :randn  && return randn(dtype, d)
    selector == :zero && return zeros(dtype, d)
    error("Specified mean initialization $(selector) is not supported.")
end


#######################
##### Scalar covariance

struct IsoRMvNormal{T} <: Distribution
    m::Array{T, 1}
    logs::Array{T, 1}
    min_s::T
    max_s::T
end

Flux.@functor IsoRMvNormal
IsoRMvNormal(m::Vector{T}, s::Vector{T}) where T = IsoRMvNormal(m, s, T(0.1), T(1.0))

function IsoRMvNormal(d::Integer; min_s=0.01, max_s=1.0, dtype::Type{<:Real}=Float32, minit::Symbol=:uniform, sinit::Symbol=:halfunit)
    m = select_rat_m(minit, dtype, d)
    logs = select_rat_logs(sinit, dtype, 1)
    return IsoRMvNormal(m, logs, dtype(min_s), dtype(max_s))
end

####
#   Functions for calculating the likelihood
####

logpdf(m::IsoRMvNormal{T}, x::Matrix{T}) where {T<:Real} = _logpdf_griso(m.m, σs(m.logs, m.min_s, m.max_s), x)
function _logpdf_griso(m::Vector{T}, s::Vector{T}, x::Matrix{T}) where {T<:Real}
    linit = -log(only(s))*size(x, 1) - T(5e-1)*size(x, 1) * log(T(2e0)*T(pi))

    l = fill(linit, 1, size(x, 2))
    s_o = only(s)
    for j in 1:size(x, 2)
        for i in 1:size(x, 1)
            @inbounds l[j] -=  T(5e-1) .* ((x[i, j] - m[i]) / s_o) ^2
        end
    end
    l
end
function logpdf_griso_back(m::Vector{T}, s::Vector{T}, x::Matrix{T}, Δy) where {T<:Real}
    Δm = zero(m)
    Δs = -sum(Δy) * size(x, 1) ./ s

    for j in 1:size(x, 2)
        for i in 1:size(x, 1)
            tmp = (x[i, j] - m[i]) / only(s)
            Δm[i] += Δy[j] * tmp / only(s)
            Δs[1] += Δy[j] * tmp^2 / only(s)
        end
    end
    Δm, Δs, NoTangent()
end
function ChainRulesCore.rrule(::typeof(_logpdf_griso), m::Vector{T}, s::Vector{T}, x::Matrix{T}) where {T<:Real}
    y = _logpdf_griso(m, s, x)
    _logpdf_pullback = Δy -> (NoTangent(), logpdf_griso_back(m, s, x, Δy)...)
    return y, _logpdf_pullback
end

####
#   Functions for generating random samples
####

Base.rand(m::IsoRMvNormal{T}, n::Int) where {T<:Real} =  (σs(m.logs, m.min_s, m.max_s) .* randn(T, length(m.m), n) .+ m.m) |> Mill.ArrayNode
Base.rand(m::IsoRMvNormal) = rand(m, 1)

####
#   Utilities
####

Base.length(m::IsoRMvNormal) = length(m.m)

######################################
######## Covariance as identity matrix

struct UnitMvNormal{T} <: Distribution
    m::Array{T, 1}
end

Flux.@functor UnitMvNormal

function UnitMvNormal(d::Integer; dtype::Type{<:Real}=Float32, minit::Symbol=:uniform)
    m = select_rat_m(minit, dtype, d)
    return UnitMvNormal(m)
end

####
#   Functions for calculating the likelihood
####
logpdf(m::UnitMvNormal{T}, x::Matrix{T}) where {T<:Real} = _logpdf_gunit(m.m, x)
function _logpdf_gunit(m::Vector{T}, x::Matrix{T}) where {T<:Real}
    linit = - T(5e-1)*size(x, 1) * log(T(2e0)*T(pi))

    l = fill(linit, 1, size(x, 2))
    for j in 1:size(x, 2)
        for i in 1:size(x, 1)
            @inbounds l[j] -=  T(5e-1) .* (x[i, j] - m[i]) ^2
        end
    end
    l
end
function logpdf_gunit_back(m::Vector{T}, x::Matrix{T}, Δy) where {T<:Real}
    Δm = zero(m)

    for j in 1:size(x, 2)
        for i in 1:size(x, 1)
            Δm[i] += (x[i, j] - m[i]) * Δy[j]
        end
    end
    Δm, NoTangent()
end

function ChainRulesCore.rrule(::typeof(_logpdf_gunit), m::Vector{T}, x::Matrix{T}) where {T<:Real}
    y = _logpdf_gunit(m, x)
    _logpdf_pullback = Δy -> (NoTangent(), logpdf_gunit_back(m, x, Δy)...)
    return y, _logpdf_pullback
end

####
#   Functions for generating random samples
####

Base.rand(m::UnitMvNormal{T}, n::Int) where {T<:Real} =  (randn(T, length(m.m), n) .+ m.m) |> Mill.ArrayNode
Base.rand(m::UnitMvNormal) = rand(m, 1)

####
#   Utilities
####

Base.length(m::UnitMvNormal) = length(m.m)

