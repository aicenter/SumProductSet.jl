using DrWatson
@quickactivate

using SumProductSet
using Flux
using Plots
using StatsBase

# generate data
f(z) = transpose(hcat(z, z + z.^2)) .+ 0.1*randn(2, length(z))
z = randn(1000)
x = f(z)

# standartize data
x = (x .- mean(x, dims=2)) ./ std(x, dims=2)
scatter(x[1,:], x[2, :])
hdim = 10

# create model
enc = Encoder(
    Dense(2, hdim, swish),
    SplitLayer(hdim, [2, 2], [identity, exp])
)
dec = Decoder(Chain(Dense(2, hdim, swish), Dense(hdim, 2, identity)))
vae = VAE(enc, dec)

opt = Adam(1e-2)
ps = Flux.params(vae)
nepoch = 700

for i in 1:nepoch
    gs = gradient(()-> -elbo(vae, x; σd=0.1), ps)            
    Flux.Optimise.update!(opt, ps, gs)
    println("Epoch $(i) ELBO:", elbo(vae, x))
end

x̂ = vae(x)

function plot_contour(m, x, title = nothing)
	xr = range(minimum(x[1,:]) - 1 , maximum(x[1,:])+ 1 , length = 50)
	yr = range(minimum(x[2,:]) - 1 , maximum(x[2,:])+ 1 , length = 50)
    p1 = contour(xr, yr, (x, y) ->  exp(reconstruct_loss(m([x, y]), [x, y])[1]))
	scatter!(p1, x[1,:], x[2,:], alpha = 0.4, label="x")
	p2 = scatter(x[1,:], x[2,:], alpha = 0.7, label="x")
	x_hat = m(x)
	scatter!(p2, x_hat[1,:], x_hat[2,:], alpha = 0.7, label="x̂")
	p = plot(p1, p2)
	!isnothing(title) && title!(p, title)
	p
end

plot_contour(vae, x)