struct Dummy{T<:Real} <: Distribution end

Dummy(_; dtype::Type{<:Real}=Float32) = Dummy{dtype}
Dummy(dtype::Type{<:Real}=Float32) = Dummy{dtype}

####
#   Functions for calculating the likelihood
####

logpdf(m::Dummy{T}, x::AbstractMatrix) where T = zeros(T, size(x, 2))
