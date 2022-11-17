using DrWatson
@quickactivate

using SumProductSet
using Flux
using Plots

f(z) = transpose(hcat(z, z + z.^2)) .+ 0.1*randn(2, length(z))
z = randn(1000)
x = f(z)

x = (x .- mean(x, dims=2)) ./ std(x, dims=2)
scatter(x[1,:], x[2, :])
hdim = 10

enc = Encoder(
    Dense(2, hdim, swish),
    SplitLayer(hdim, [2, 2], [identity, exp])
)

dec = Decoder(Chain(Dense(2, hdim, swish), Dense(hdim, 2, identity)))

vae = VAE(enc, dec)

opt = Adam(1e-2)
ps = Flux.params(vae)

for i in 1:700
    gs = gradient(()-> -elbo(vae, x; σd=0.1), ps)            
    Flux.Optimise.update!(opt, ps, gs)
    println("Iter $(i) ELBO:", elbo(vae, x))
end

x̂ = vae(x)

display(scatter!(x̂[1, :], x̂[2, :]))

function plot_contour(m, x, title = nothing)
	xr = range(minimum(x[1,:]) - 1 , maximum(x[1,:])+ 1 , length = 50)
	yr = range(minimum(x[2,:]) - 1 , maximum(x[2,:])+ 1 , length = 50)
    p1 = contour(xr, yr, (x, y) ->  exp(reconstruct_loss(m([x, y]), [x, y])[1]))
	p2 = deepcopy(p1)
	scatter!(p2, x[1,:], x[2,:], alpha = 0.4)
	p = plot(p1, p2)
	!isnothing(title) && title!(p, title)
	p
end

plot_contour(vae, x)