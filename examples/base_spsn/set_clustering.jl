using DrWatson
@quickactivate
using SumProductSet
using Plots
using Flux
using StatsBase
using Clustering
import Mill


### function to train the model
function train!(m, x; niter::Int=200, opt=ADAM(0.02))
    ps = Flux.params(m)

    loss = () -> em_loss(m, x)
    for i in 1:niter
        println("Iter $(i) ll: $(mean(logpdf(m, x)))")
        gs = gradient(loss, ps)
        Flux.Optimise.update!(opt, ps, gs)
    end
    println("Final ll: $(mean(logpdf(m, x)))")
end

### Parameters of known model
μ = ([4., 6.], [10., 9.], [10., 9.])
Σ = ([3. -1.6; -1.6 3], [3. 1.6; 1.6 6], [3. 1.6; 1.6 6])
λ = (26, 26, 6)


### Contructor of known model
function knownsetmixture(μs, Σs, λs, prior)
    components = map(zip(μs, Σs, λs)) do ps
        pc = Poisson(log(ps[3]))
        pf = MvNormalParams(Float32.(ps[1]), Float32.(ps[2]))
        SetNode(pf, pc)
    end
    SumNode(components, prior)    
end
### Creating and sampling known model
m1 = knownsetmixture(μ, Σ, λ, [1f0, 1, 1])
nbags = 300
bags, baglabels = randwithlabel(m1, nbags)

instances = bags.data.data
bagids = bags.bags

# Normalize data
mn = mean(instances, dims=2)
sd =  std(instances, dims=2)
instances = (instances .- mn) ./ sd

stdbags = Mill.BagNode(instances, bagids)

instlabels = mapreduce(a -> repeat([a[1]], length(a[2])), vcat, zip(baglabels, bagids))
display(scatter(instances[1, :], instances[2, :], group=instlabels, alpha=0.8, title="Normalized instance space"))

### Create and train mixure model
m2 = setmixture(3, 1, 2)
train!(m2, stdbags)
clusters = mapslices(argmax, logjnt(m2, stdbags), dims=1)[:]
ari = randindex(baglabels, clusters)[1]
println("Achieved ARI: $(ari) on train set.")

### Generate testing data
nbags = 300
bags_tst, baglabels_tst = randwithlabel(m1, nbags)

instances = bags_tst.data.data
bagids = bags_tst.bags
instances = (instances .- mn) ./ sd
stdbags_tst = Mill.BagNode(instances, bagids)

clusters_tst = mapslices(argmax, logjnt(m2, stdbags_tst), dims=1)[:]  
ari_tst = randindex(baglabels_tst, clusters_tst)[1] 
println("Achieved ARI: $(ari_tst) on test set.")

