using Flux, Graphs, Statistics, Plots, GraphPlot, Zygote
using JSON
using JsonGrinder
using Cairo, Compose
using SumProductSet
import Distributions
import Mill

n_samples, n_val = 3000, 4774

data_file = "data/recipes_train.json"
samples = open(data_file,"r") do fid
	Vector{Dict}(JSON.parse(read(fid, String)))
end
JSON.print(samples[1], 2)

sch = JsonGrinder.schema(samples[1:n_samples])
delete!(sch.childs,:id)
extractor = suggestextractor(sch)

extract_data = ExtractDict(deepcopy(extractor.dict))
extract_target = ExtractDict(deepcopy(extractor.dict))
delete!(extract_target.dict, :ingredients)
delete!(extract_data.dict, :cuisine)

data = extract_data.(samples[1:n_samples])
data = reduce(Mill.catobs, data)
data = data[:ingredients]

target = extract_target.(samples[1:n_samples])
target = reduce(Mill.catobs, target)[:cuisine].data
target = Flux.onecold(target)

m = reflectinmodel(data, maximum(target))

predict = x->mapslices(argmax, logjnt(m, x), dims=1)[:] 
accuracy(y, x) = mean(y .== predict(x))

function train!(m, x, y; niter::Int=200, opt=ADAM(0.02))
    ps = Flux.params(m)

    loss = () -> -mean( logjnt(m, x)[CartesianIndex.(y, 1:length(y))])
    for i in 1:niter
        println("Iter $(i) ll: $(mean(logpdf(m, x))), acc: $(accuracy(y, x))")
        gs = gradient(loss, ps)
        Flux.Optimise.update!(opt, ps, gs)
    end
    println("Final ll: $(mean(logpdf(m, x))), acc: $(accuracy(y, x))")
end