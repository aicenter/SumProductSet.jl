using DrWatson
@quickactivate
using SumProductSet
using Plots
using Random
using BSON

import Mill

μ = ([4., 6.], [10., 9.], [10., 9.])
Σ = ([3. -1.6; -1.6 3], [3. 1.6; 1.6 6], [3. 1.6; 1.6 6])
λ = (26, 26, 6)

seeds = 1:20

savedir = "/home/martin/datasets/clean/toy_pp/"

function knownsetmixture(μs, Σs, λs, prior)
    components = map(zip(μs, Σs, λs)) do ps
        pc = _Poisson(log(ps[3]))
        pf = _MvNormalParams(ps[1], ps[2])
        SetNode(pf, pc)
    end
    SumNode(components, prior)    
end
m1 = knownsetmixture(μ, Σ, λ, [1., 1., 1.])
nbags = 200

for seed in 1:length(seeds)
    Random.seed!(seed)
    bagnode, baglabels = randwithlabel(m1, nbags)
    data = bagnode.data.data
    labs = baglabels
    bags = bagnode.bags.bags
    # scatteredbags = mapreduce(a -> repeat([a[1]], length(a[2])), vcat, zip(1:nbags, bagids))
    @show size(data)
    bson("$(savedir)/iidcluster_$(seed).bson", data=data, labs=labs, bags=bags)
end