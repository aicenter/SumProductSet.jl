using DrWatson
@quickactivate
using SumProductSet
using Plots
using Flux
using StatsBase
using Clustering
import Mill

### function to train model
# inefficient implementation of loss for now
function train!(m, x, y; niter::Int=200, opt=ADAM(0.02))
    ps = Flux.params(m)

    loss = () -> sl_loss(m, x, y)
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
        pc = _Poisson(log(ps[3]))
        pf = _MvNormalParams(ps[1], ps[2])
        SetNode(pf, pc)
    end
    SumNode(components, prior)    
end

### Creating and sampling known model
m1 = knownsetmixture(μ, Σ, λ, [1., 1., 1.])
nbags = 100
bags, baglabels = randwithlabel(m1, nbags)

instances = bags.data.data
bagids = bags.bags

# Normalize data
mn = mean(instances, dims=2)
sd =  std(instances, dims=2)
instances = (instances .- mn) ./ sd

stdbags = Mill.BagNode(Mill.ArrayNode(instances), bagids)

instlabels = mapreduce(a -> repeat([a[1]], length(a[2])), vcat, zip(baglabels, bagids))
display(scatter(instances[1, :], instances[2, :], group=instlabels, alpha=0.8, title="Normalized instance space"))

### Create and train mixure model
m2 = setmixture(3, 1, 2)
train!(m2, stdbags, baglabels)
predtrn = mapslices(argmax, logjnt(m2, stdbags), dims=1)[:]

acc(prediction, target) = mean(prediction .== target)    
println("Train data acc: $(acc(predtrn, baglabels))")

# Generate testing data
nbags = 300
bags_tst, baglabels_tst = randwithlabel(m1, nbags)

instances = bags_tst.data.data
bagids = bags_tst.bags

mn = mean(instances, dims=2)
sd =  std(instances, dims=2)
instances = (instances .- mn) ./ sd

stdbags_tst = Mill.BagNode(Mill.ArrayNode(instances), bagids)

predtst = mapslices(argmax, logjnt(m2, stdbags_tst), dims=1)[:]   
println("Test data acc: $(acc(predtst, baglabels_tst))")
