# define split layer
struct SplitLayer
    μ::Flux.Dense
    σ::Flux.Dense
end

Flux.@functor SplitLayer

function SplitLayer(in::Int, out::Vector{Int}, acts::Vector)
	SplitLayer(
		Flux.Dense(in, out[1], acts[1]),
		Flux.Dense(in, out[2], acts[2])
	)
end
(m::SplitLayer)(x::AbstractArray) = (m.μ(x), m.σ(x))

# define encoder
struct Encoder{B, S<:SplitLayer}
    body::B
    split_layer::S 
end
Flux.@functor Encoder
(m::Encoder)(x::AbstractArray) = m.split_layer(m.body(x))

# define decoder
struct Decoder{B}
    body::B
end
Flux.@functor Decoder
(m::Decoder)(x::AbstractArray) = m.body(x)

struct VAE{E<:Encoder, D<:Decoder}
    encoder::E
    decoder::D
end
Flux.@functor VAE
function (m::VAE)(x::AbstractArray{T}) where T
    μ, σ = m.encoder(x)
    z = μ .+ σ .* randn(T, size(μ)...)
    m.decoder(z)
end

# KL divergence between N(μ, diag(σ^2)) and N(0, I)
kl_loss(μ::Array{T, 2}, σ::Array{T, 2}) where T = T(0.5) .* sum( σ.^2 .+ μ.^2 .- T(1) .- T(2).*log.(σ), dims=1)

function reconstruct_loss(x::AbstractArray{T}, x̂::AbstractArray{T}; σ::T=T(0.5)) where T 
    k = size(x, 1)
    y = (x .- x̂) ./ σ
    -T(k/2) * ( T(log(2*pi)) + 2*log(σ)) .- T(0.5)*sum( y.^2, dims=1)
end

function elbo(m::VAE, x::AbstractArray{T}; σd=0.1) where T
    
    # sample latent variable
    μ, σ = m.encoder(x) # encoder contains split layer
    # reparametrisation trick
    z = μ .+ σ .* randn(T, size(μ)...)

    rec = reconstruct_loss(x, m.decoder(z); σ=σd)
    kl = kl_loss(μ, σ)
    mean(rec - kl)
end

# this is not actual logpdf but rather ELBO
logpdf(m::VAE, x::AbstractArray) = elbo(m, x) 
    