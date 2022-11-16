using DrWatson
@quickactivate

using SumProductSet
using Flux
using Plots

f(z) = transpose(hcat(z, z .+ z.^2)) .+ 0.1*randn(2, length(z))
z = randn(1000)
x = f(z)
scatter(x[1,:], x[2, :])

enc = Encoder(
    Dense(2, 2, swish),
    SplitLayer(2, [2, 2], [identity, exp])
)

dec = Decoder(Chain(Dense(2, 2, swish), Dense(2, 2, identity)))

vae = VAE(enc, dec)

opt = Adam(1e-2)
ps = Flux.params(vae)

for i in 1:2000
    gs = gradient(()-> -elbo(vae, x), ps)            
    Flux.Optimise.update!(opt, ps, gs)
    println("Iter $(i) ELBO:", elbo(vae, x))
end

x̂ = vae(x)

scatter!(x̂[1, :], x̂[2, :])