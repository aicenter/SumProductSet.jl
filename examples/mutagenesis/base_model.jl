using DrWatson
@quickactivate 

using JsonGrinder, Flux, MLDatasets, Statistics, Random, Printf, JSON3, HierarchicalUtils
using SumProductSet
import Mill

train_data = MLDatasets.Mutagenesis(split=:train)
x_train, y_train = train_data.features, train_data.targets
y_train .+= 1;
sch = JsonGrinder.schema(x_train)
extractor = suggestextractor(sch)
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
        gs = gradient(() -> SumProductSet.disc_loss(m, x, y), ps)
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
n_class = length(unique(y_train))

m = reflectinmodel(ds_train[1], 2)  # creates model with default hyperparameters
cb = i -> status(i, ds_train, y_train, ds_test, y_test)
train!(m, ds_train, y_train; niter=1, opt=ADAM(0.2), cb=cb)
@time train!(m, ds_train, y_train; niter=100, opt=ADAM(0.2), cb=cb)

sum(length, Flux.params(m))

printtree(m, htrunc=25, vtrunc=25)
