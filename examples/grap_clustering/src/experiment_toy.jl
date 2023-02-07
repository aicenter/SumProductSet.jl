using Flux, Graphs, Statistics, Plots, GraphRecipes, Zygote
using SumProductSet
import Mill

g = SimpleGraph(11)
for e in [(1, 2), (1, 3), (1, 4),
        (2, 4), (2, 5),
        (3, 4), (3, 5), (3, 6),
        (4, 5), (4, 6),

        (8, 9), (8, 7), (8, 10),
        (9, 7), (9, 6),
        (10, 7),
        (11, 8), (11, 9), (11, 10)
    ]
    add_edge!(g, e...)
end

vs = collect(vertices(g))
x = Flux.onehotbatch(vs, vs)
bn = Mill.BagNode(x, g.fadjlist)
m = reflectinmodel(bn[1], 2)

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
predict = x->mapslices(argmax, logjnt(m, x), dims=1)[:] 
clusters = predict(bn)
display(graphplot(g, marker=colors[clusters]))

train!(m, bn; opt=ADAM(0.02))

colors = [colorant"yellow", colorant"red"]
clusters = predict(bn)
display(graphplot(g, marker=colors[clusters]))