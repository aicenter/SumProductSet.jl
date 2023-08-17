using DrWatson
@quickactivate 

using JsonGrinder, Flux, MLDatasets, Statistics, Random, Printf, JSON3, HierarchicalUtils
using SumProductSet
import Mill

function default_scalar_extractor()
    [
    (e -> length(keys(e)) <= 100 && JsonGrinder.is_numeric_or_numeric_string(e),
        (e, uniontypes) -> ExtractCategorical(keys(e), uniontypes)),
    (e -> JsonGrinder.is_intable(e),
        (e, uniontypes) -> extractscalar(Int32, e, uniontypes)),
    (e -> JsonGrinder.is_floatable(e),
        (e, uniontypes) -> extractscalar(Float32, e, uniontypes)),
    (e -> (keys_len = length(keys(e)); keys_len / e.updated < 0.1 && keys_len < 10000 && !JsonGrinder.is_numeric_or_numeric_string(e)),
        (e, uniontypes) -> ExtractCategorical(keys(e), uniontypes)),
    (e -> true,
        (e, uniontypes) -> ExtractScalar(Float32, 0., 1., false)),]
end

train_data = MLDatasets.Mutagenesis(split=:train)
x_train, y_train = train_data.features, train_data.targets
y_train .+= 1;
sch = JsonGrinder.schema(x_train)
extractor = suggestextractor(sch, (; scalar_extractors = default_scalar_extractor()))
ds_train = Mill.catobs(extractor.(x_train))

printtree(sch, htrunc=25, vtrunc=25)

test_data = MLDatasets.Mutagenesis(split=:test)
x_test, y_test = test_data.features, test_data.targets
y_test .+= 1;
ds_test = Mill.catobs(extractor.(x_test));

function train!(m, x, y; niter::Int=100, opt=ADAM(0.1), cb=iter->())
    ps = Flux.params(m)
    cb(0)
    for i in 1:niter
        gs = gradient(() -> SumProductSet.ce_loss(m, x, y), ps)
        Flux.Optimise.update!(opt, ps, gs)
        cb(i)
    end
end

predict = x-> Flux.onecold(softmax(logjnt(m, x)))

accuracy(y, x) = mean(y .== predict(x))
function status(iter, x_trn, y_trn, x_tst, y_tst)
    acc_trn = accuracy(y_trn, x_trn) 
    acc_tst = accuracy(y_tst, x_tst)
    
    @printf("Epoch %i - acc: | %.3f  %.3f | \n", iter, acc_trn, acc_tst)
end

Random.seed!(1234);
dir_rand(d) = (r = rand(d); return r ./ sum(r))
f_cat = d->Categorical(log.(dir_rand(d))) # choose how to represent categorical variables
f_cont = d->gmm(2, d)  # choose how to represent continuous variables

m = reflectinmodel(ds_train[1], 2)
cb = i -> status(i, ds_train, y_train, ds_test, y_test)
@time train!(m, ds_train, y_train; niter=100, opt=ADAM(0.2), cb=cb)

sum(length, Flux.params(m))

printtree(m, htrunc=25, vtrunc=25)