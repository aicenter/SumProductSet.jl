using JsonGrinder, Flux, MLDatasets, Statistics, Random
using SumProductSet
import Mill

Random.seed!(42)

function train!(m, x, y; niter::Int=10, opt=ADAM(0.02))
    ps = Flux.params(m)

    loss = () -> -mean( logjnt(m, x)[CartesianIndex.(y, 1:length(y))])
    for i in 1:niter
        println("Iter $(i) ll: $(mean(logpdf(m, x))), acc: $(accuracy(y, x))")
        gs = gradient(loss, ps)
        Flux.Optimise.update!(opt, ps, gs)
    end
    println("Final ll: $(mean(logpdf(m, x))), acc: $(accuracy(y, x))")
end

# redefinition of standard scalar extractor
function default_scalar_extractor()
	[
	(e -> length(keys(e)) <= 100 && JsonGrinder.is_numeric_or_numeric_string(e),
		(e, uniontypes) -> ExtractCategorical(keys(e), uniontypes)),
	(e -> JsonGrinder.is_intable(e),
		(e, uniontypes) -> extractscalar(Int64, e, uniontypes)),
	(e -> JsonGrinder.is_floatable(e),
	 	(e, uniontypes) -> extractscalar(Float64, e, uniontypes)),
	(e -> (keys_len = length(keys(e)); keys_len / e.updated < 0.1 && keys_len < 10000 && !JsonGrinder.is_numeric_or_numeric_string(e)),
		(e, uniontypes) -> ExtractCategorical(keys(e), uniontypes)),
	(e -> true,
		(e, uniontypes) -> ExtractScalar(Float64, 0., 1., false)),]
end

# get train data; it will be downloaded in fisrt run
x_train, y_train = MLDatasets.Mutagenesis.traindata();
sch = JsonGrinder.schema(x_train)
extractor = suggestextractor(sch, (; scalar_extractors = default_scalar_extractor()))
ds_train = Mill.catobs(extractor.(x_train))
# remap labels from [0, 1] to [1, 2]
y_train .+= 1

m = reflectinmodel(ds_train[1], 2)

# create predict function
predict = x->mapslices(argmax, logjnt(m, x), dims=1)[:] 
accuracy(y, x) = mean(y .== predict(x))

# train the model
train!(m, ds_train, y_train; niter=10, opt=ADAM(0.1))

# get test data
x_test, y_test = MLDatasets.Mutagenesis.testdata();
y_test .+= 1
ds_test = Mill.catobs(extractor.(x_test))


@show accuracy(y_train, ds_train)
@show accuracy(y_test, ds_test)

