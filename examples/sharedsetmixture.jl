using DrWatson
@quickactivate
using SumProductSet
using Plots
using Flux
using StatsBase
using Clustering
import Mill

function train!(m::SumNode, x::Mill.BagNode; niter::Int=300, opt=ADAM(0.01))
    ps = Flux.params(m)

    for i in 1:niter
        println("Iter $(i) ll: $(mean(logpdf(m, x)))")
        gs = gradient(()->-mean(logpdf(m, x)), ps)
        Flux.Optimise.update!(opt, ps, gs)
    end
    println("Final ll: $(mean(logpdf(m, x)))")
end

λ = [6, 6]
μ = [[0., 0.], [15., 15.]]
Σ = [[3. -1.6; -1.6 3], [3. -1.6; -1.6 3]]

sμ = [[7.5, 7.5]]
sΣ = [[3. 1.6; 1.6 6]]

function knownsharedsetmixture(sμ, sΣ, μ, Σ, λ, prior)
    sharedcomps = [_MvNormalParams(sμi, sΣi) for (sμi, sΣi) in zip(sμ, sΣ)]

    bagcomps = map(zip(μ, Σ, λ)) do (μi, Σi, λi)
        nonsharedcomps = [_MvNormalParams(μi, Σi)]
        pc = _Poisson(log(λi))
        pf = SumNode([nonsharedcomps; sharedcomps])
        SetNode(pf, pc)
    end
    SumNode(bagcomps, prior)    
end

m1 = knownsharedsetmixture(sμ, sΣ, μ, Σ, λ, [1., 1.])
nbags = 300
bags, baglabels = randwithlabel(m1, nbags)

instances = bags.data.data
bagids = bags.bags

mn = mean(instances, dims=2)
sd =  std(instances, dims=2)
instances = (instances .- mn) ./ sd

stdbags = Mill.BagNode(Mill.ArrayNode(instances), bagids)

instlabels = mapreduce(a -> repeat([a[1]], length(a[2])), vcat, zip(baglabels, bagids))
display(scatter(instances[1, :], instances[2, :], group=instlabels, alpha=0.8, title="Normalized instance space"))

m2 = sharedsetmixture(2, 1, 1, 2)
train!(m2, stdbags)
clusters = mapslices(argmax, logjnt(m2, stdbags), dims=1)[:]
ari = randindex(baglabels, clusters)[1]
println("Achieved ARI: $(ari) on train set.")
