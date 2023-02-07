using Flux, Graphs, Statistics, Plots, GraphPlot, Zygote
using Cairo, Compose
using SumProductSet
import Distributions
import Mill

# x = readlines("data/out.ucidata-zachary");
# e = map(yi-> tuple(parse.(Int, split(yi, " "))...), x[3:end])
g = smallgraph(:karate)
# map(ei->add_edge!(g, ei), e)


vs = collect(vertices(g))
x = Flux.onehotbatch(vs, vs)
bn = Mill.BagNode(x, g.fadjlist)
dir_rand = d->rand(Distributions.Dirichlet(d, 5*d))
f_cat = d->_Categorical(log.(dir_rand(d)))
m = reflectinmodel(bn[1], 2; f_card=()->_Poisson(log(3)), f_cat=f_cat)

function ul_loss(m::SumNode, xu)
    # E-step for unlabeled data
    p = []
    Zygote.ignore() do 
        p = softmax(logjnt(m, xu); dims=1)
    end
    -mean(p .* logjnt(m, xu))
end 

function train!(m, x; niter::Int=1000, opt=ADAM(1.))
    ps = Flux.params(m)

    loss = () -> ul_loss(m, x)
    for i in 1:niter
        println("Iter $(i) ll: $(mean(logpdf(m, x)))")
        gs = gradient(loss, ps)
        Flux.Optimise.update!(opt, ps, gs)
    end
    println("Final ll: $(mean(logpdf(m, x)))")
end
colors = [colorant"yellow", colorant"red"]
nodelabel = 1:nv(g)

predict = x->mapslices(argmax, logjnt(m, x), dims=1)[:] 
clusters = predict(bn)
draw(PDF("karate_1.pdf", 16cm, 16cm), gplot(g, nodefillc=colors[clusters], nodelabel=nodelabel))

train!(m, bn; opt=ADAM(0.1))

clusters = predict(bn)
 draw(PDF("karate_2.pdf", 16cm, 16cm), gplot(g, nodefillc=colors[clusters], nodelabel=nodelabel))