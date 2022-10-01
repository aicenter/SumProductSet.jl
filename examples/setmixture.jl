using DrWatson
@quickactivate
using SumProductSet
using Plots
using Flux
using StatsBase
import Mill

function train!(m::SumNode, x::Mill.BagNode; niter::Int=400, opt=ADAM(0.02))
    ps = Flux.params(m)

    for i in 1:niter
        println("Iter $(i) ll: $(mean(logpdf(m, x)))")
        gs = gradient(()->-mean(logpdf(m, x)), ps)
        Flux.Optimise.update!(opt, ps, gs)
    end
    println("Final ll: $(mean(logpdf(m, x)))")
end

μ = ([4., 6.], [10., 9.], [10., 9.])
Σ = ([3. -1.6; -1.6 3], [3. 1.6; 1.6 6], [3. 1.6; 1.6 6])
λ = (26, 26, 6)

function knownsetmixture(μs, Σs, λs)
    components = map(zip(μs, Σs, λs)) do ps
        pc = _Poisson(log(ps[3]))
        pf = _MvNormalParams(ps[1], ps[2])
        SetNode(pf, pc)
    end
    SumNode(components)    
end
m1 = knownsetmixture(μ, Σ, λ)
nbags = 300
bags, baglabels = randwithlabel(m1, nbags)

instances = bags.data.data
bagids = bags.bags

instlabels = mapreduce(a -> repeat([a[1]], length(a[2])), vcat, zip(baglabels, bagids))
display(scatter(instances[1, :], instances[2, :], group=instlabels, alpha=0.8, title="Instance space"))

println("Reference average loglikelihood: $(mean(logpdf(m1, bags)))")
m2 = setmixture(2, 1, 2; fdist=:MvNormal)
train!(m2, bags)
