#!/usr/bin/env sh
#SBATCH --array=1-150
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1
#SBATCH --partition=cpu
#SBATCH --out=/home/rektomar/logs/hmill_mip/%x-%j.out
#= 
srun julia scripts/hmillclassifier.jl --n $SLURM_ARRAY_TASK_ID --m $1
exit
# =#

using DrWatson
@quickactivate

import Mill
using Flux
using NNlib
using Random
using Printf
using ArgParse
using StatsBase
using Clustering
using DataFrames
using BSON: @load
using PrettyTables
using LinearAlgebra
using EvalMetrics

function hmill_classifier(x_trn; hdim=16, cdim=8, activation=:relu, aggregation=:SegmentedMeanMax, nclasses=2, nlayers=1, seed=nothing)
    isnothing(seed) ? nothing : Random.seed!(seed)
    activation = eval(activation)
    aggregation = Mill.BagCount ∘ eval(Expr(:., :Mill, QuoteNode(aggregation)))

    extractor = Mill.reflectinmodel(x_trn, d -> Dense(d, hdim, activation), aggregation)
    
    if nlayers == 1
        classifier = Dense(hdim, nclasses)
    elseif nlayers == 2
        classifier = Chain(Dense(hdim, cdim, activation), Dense(cdim, nclasses))
    elseif nlayers == 3
        classifier = Chain(Dense(hdim, cdim, activation), Dense(cdim, cdim, activation), Dense(cdim, nclasses))
    else
        error("Wrong number of classifier layers")
    end

    Chain(extractor, classifier)
end

const maxseed = 20
const path = "/home/$(ENV["USER"])/datasets/clean/mill"

function train!(m, x_trn::Mill.BagNode, y_trn::Vector{Int}; cb=()->(), niter::Int=200, opt=ADAM(0.02))

    ps = Flux.params(m)
    loss(ds, y_oh) = Flux.logitcrossentropy(m(ds), y_oh)

    status = cb()

    for _ in 1:niter
        gs = gradient(()-> loss(x_trn, Flux.onehotbatch(y_trn, 1:2)), ps)            
        Flux.Optimise.update!(opt, ps, gs)
        status = cb()
    end

    return status
end

acc(prediction, target) = mean(prediction .== target)    

function status!(m, x_trn, x_val, x_tst, y_trn, y_val, y_tst, verbose=false)

    ŷ_trn = mapslices(argmax, m(x_trn), dims = 1)[:]
    ŷ_val = mapslices(argmax, m(x_val), dims = 1)[:]
    ŷ_tst = mapslices(argmax, m(x_tst), dims = 1)[:]
    loss(ds, y_oh) = Flux.logitcrossentropy(m(ds), y_oh)

    l_trn = loss(x_trn, Flux.onehotbatch(y_trn, 1:2))
    l_val = loss(x_val, Flux.onehotbatch(y_val, 1:2))
    l_tst = loss(x_tst, Flux.onehotbatch(y_tst, 1:2))

    ari_trn = randindex(y_trn, ŷ_trn)[1]
    ari_val = randindex(y_val, ŷ_val)[1]
    ari_tst = randindex(y_tst, ŷ_tst)[1]
    ri_trn = randindex(y_trn, ŷ_trn)[2]
    ri_val = randindex(y_val, ŷ_val)[2]
    ri_tst = randindex(y_tst, ŷ_tst)[2]

    acc_trn = acc(ŷ_trn, y_trn)
    acc_val = acc(ŷ_val, y_val)
    acc_tst = acc(ŷ_tst, y_tst)

    # need to convert labels [1, 2] to [0, 1]
    mcc_trn = EvalMetrics.mcc(ConfusionMatrix(ŷ_trn .- 1, y_trn .- 1))
    mcc_val = EvalMetrics.mcc(ConfusionMatrix(ŷ_val .- 1, y_val .- 1))
    mcc_tst = EvalMetrics.mcc(ConfusionMatrix(ŷ_tst .- 1, y_tst .- 1))

    @printf("loss:| %2.4e %2.4e %2.4e |   ri:| %.2f %.2f %.2f |  ari:| %.2f %.2f %.2f |  acc:| %.2f %.2f %.2f |   mcc:| %.2f %.2f %.2f \n",
        l_trn, l_val, l_tst, ri_trn, ri_val, ri_tst, ari_trn, ari_val, ari_tst, acc_trn, acc_val, acc_tst, mcc_trn, mcc_val, mcc_tst)

    (; l_trn, l_val, l_tst, ari_trn, ari_val, ari_tst, ri_trn, ri_val, ri_tst, acc_trn, acc_val, acc_tst, mcc_trn, mcc_val, mcc_tst)
end



dirdata = "mill"
datasets = [
    (dataset="brown_creeper",              ndims=38,    ndata=10232,   nclass=2,    nbags=548 ) # end=10232    1
    (dataset="corel_african",              ndims=9,     ndata=7947,    nclass=2,    nbags=2000) # end=7947     2
    (dataset="corel_beach",                ndims=9,     ndata=7947,    nclass=2,    nbags=2000) # end=7947     3
    (dataset="elephant",                   ndims=230,   ndata=1391,    nclass=2,    nbags=200 ) # end=1391     4
    (dataset="fox",                        ndims=230,   ndata=1320,    nclass=2,    nbags=200 ) # end=1320     5
    (dataset="musk_1",                     ndims=166,   ndata=476,     nclass=2,    nbags=92  ) # end=476      6
    (dataset="musk_2",                     ndims=166,   ndata=6598,    nclass=2,    nbags=102 ) # end=6598     7
    (dataset="mutagenesis_1",              ndims=7,     ndata=10486,   nclass=2,    nbags=188 ) # end=10486    8
    (dataset="mutagenesis_2",              ndims=7,     ndata=2132,    nclass=2,    nbags=42  ) # end=2132     9
    (dataset="newsgroups_1",               ndims=200,   ndata=5443,    nclass=2,    nbags=100 ) # end=5443     10
    (dataset="newsgroups_2",               ndims=200,   ndata=3094,    nclass=2,    nbags=100 ) # end=3094     11
    (dataset="newsgroups_3",               ndims=200,   ndata=5175,    nclass=2,    nbags=100 ) # end=5175     12
    (dataset="protein",                    ndims=9,     ndata=26611,   nclass=2,    nbags=193 ) # end=26611    13
    (dataset="tiger",                      ndims=230,   ndata=1220,    nclass=2,    nbags=200 ) # end=1220     14
    (dataset="ucsb_breast_cancer",         ndims=708,   ndata=2002,    nclass=2,    nbags=58  ) # end=2002     15
    # (dataset="web_1",                      ndims=5863,  ndata=2212,    nclass=2,    nbags=75  ) # end=2212     16
    # (dataset="web_2",                      ndims=6519,  ndata=2219,    nclass=2,    nbags=75  ) # end=2219     17
    # (dataset="web_3",                      ndims=6306,  ndata=2514,    nclass=2,    nbags=75  ) # end=2514     18
    # (dataset="web_4",                      ndims=6059,  ndata=2291,    nclass=2,    nbags=75  ) # end=2291     19
    (dataset="winter_wren",                ndims=38,    ndata=10232,   nclass=2,    nbags=548 ) # end=10232    20
]

function command_line()
    s = ArgParseSettings()
    @add_arg_table s begin
        ("--n"; arg_type = Int; default=1);
        ("--m"; arg_type = Int; default=1);
    end
    parse_args(s)
end
function experiment(id::Int, dirdata::String, dataset_attributes::NamedTuple, grid)
    (; dataset, ndims, ndata, nbags) = dataset_attributes

    mtype = :hmill_clf
    hdim, nlayers, aggregation, activation, nepoch, split, seed = collect(grid)[id]
    Random.seed!(seed)

    (; dirdata, dataset, ndims, ndata, nbags, hdim, nlayers, aggregation, activation, nepoch, split, seed, mtype, itype=Int64, ftype=Float64, ngrid=length(grid))
end
function preprocess(x::AbstractArray{Tr,2}, y::AbstractArray{Ti,1}, b::AbstractArray{Ti,1}, i::AbstractArray{Ti,1}=collect(1:maximum(b)), split::AbstractArray{Tr,1}=Tr.([64e-2, 16e-2, 2e-1])) where {Tr<:Real,Ti<:Int}

    xbn = Mill.BagNode(x, b)
    n = cumsum(map(n->ceil(Ti, n), split*nobs(xbn)))

    # filter feature dimensions with low variance on train data
    x_trn_inst = xbn[i[1:n[1]]].data.data
    mn = mean(x_trn_inst, dims=2)[:]
    sd = std(x_trn_inst, dims=2)[:]
    mask = sd .> Tr(1e-5)

    # standartize data
    x = (x[mask, :] .- mn[mask]) ./ sd[mask]
    x = Mill.BagNode(x, b)  

    # created train/val/test split
    x_trn = x[i[1:n[1]]]
    x_val = x[i[n[1]+1:n[2]]]
    x_tst = x[i[n[2]+1:end]]

    # get labels corresponding to each split
    y_trn = map(j->y[j], x.bags[i[1:n[1]]])
    y_val = map(j->y[j], x.bags[i[n[1]+1:n[2]]])
    y_tst = map(j->y[j], x.bags[i[n[2]+1:end]])

    # get bag labels
    y_trn = map(y->maximum(y), y_trn)
    y_val = map(y->maximum(y), y_val)
    y_tst = map(y->maximum(y), y_tst)

    return x_trn, x_val, x_tst, y_trn, y_val, y_tst
end
function generate_real_data(config::NamedTuple)
    (; dataset, ndata, nbags, ftype, itype) = config

    @load "$(path)/$(dataset).bson" data labs bags

    x = Matrix{ftype}(data)
    y = Vector{itype}(labs)
    b = Vector{itype}(bags)

    Random.seed!(1)
    i = randperm(nbags)

    return ntuple2dict((; dataset, ndims, ndata, x, y, b, i))
end
function load_real_data(config::NamedTuple)
    (; ftype, split, seed) = config
    file, _ = produce_or_load(datadir("hmill_mip/datasets"),
                              config,
                              generate_real_data,
                              suffix="bson",
                              sort=false,
                              accesses=(:dataset, :ndims, :nbags, :ndata))
    @unpack x, y, b, i = file
    data = preprocess(ftype.(x), y, b, i, ftype.(split))
    return data..., config
end

function estimate(config::NamedTuple)
    (; dataset,  hdim, nlayers, aggregation, activation, nepoch, seed) = config
    x_trn, x_val, x_tst, y_trn, y_val, y_tst, config = load_real_data(config)

    @show dataset, hdim, nlayers, aggregation, activation, seed

    model = hmill_classifier(x_trn; activation=activation, aggregation=aggregation, nlayers=nlayers, hdim=hdim, cdim=hdim, seed=seed)

    status = train!(model, x_trn, y_trn; cb=()->status!(model, x_trn, x_val, x_tst, y_trn, y_val, y_tst), niter=nepoch)

    ntuple2dict(merge(config, status, (; model)))
end

function result_table(; show=[:l, :ri, :ari, :acc], type=[:trn, :val, :tst], kwargs...)
    df = collect_results(datadir("hmill_mip/results"); kwargs...)
    df = groupby(df, [:dataset, :m, :n, :mtype, :cardtype])

    table_metrics = [Symbol(metric, :_, settype) for metric in show for settype in type]
    table_operations = [op => mean for op in table_metrics]

    df = combine(df, table_operations..., renamecols=false)
    df = combine(df, :dataset, :n, :m, :cardtype, [xi => ByRow(n->round(n, sigdigits=3)) for xi in table_metrics]..., renamecols=false)
end

function main_local()
    Random.seed!(1)

    @load "$(path)/fox.bson" data labs bags
    perm = randperm(maximum(bags))
    x_trn, x_val, x_tst, y_trn, y_val, y_tst = preprocess(Float64.(data), labs, bags, perm)

    niter = 100
    model = hmill_classifier(x_trn)

    train!(model, x_trn, y_trn; niter=niter, cb=()->status!(model, x_trn, x_val, x_tst, y_trn, y_val, y_tst))

    nothing
end

function main_slurm()
    @unpack n, m = command_line()
    dataset = datasets[m]
    grid = Iterators.product(
        [8, 16, 32, 64, 128],
        [1, 2, 3],
        [:SegmentedMeanMax],
        [:tanh, :relu],
        [1000],
        [[64e-2, 16e-2, 2e-1]],
        collect(1:5))

        # hdim, cdim, aggregation, activation, nepoch, seeds
        # |grid| = 5 * 3 * 1 * 2 * 5 = 150

    produce_or_load(datadir("hmill_mip/results"),
                    experiment(n, dirdata, dataset, grid),
                    estimate;
                    suffix="jld2",
                    sort=false,
                    ignores=(:dirdata, :ngrid),
                    verbose=false)
end


# main_local()
main_slurm()

# Base.run(`clear`)
# best_architecture_table(find_best_architecture(; s=:l_val, x=:l_tst); x=:l_tst)
# best_architecture_table(find_best_architecture(; s=:l_val, x=:i_tst); x=:i_tst)
# best_architecture_table(find_best_architecture(; s=:l_val, x=:r_tst); x=:r_tst)

# best_architecture_table(find_best_architecture(; s=:l_val, x=:l_tst, rexclude=[r"ILM"]); x=:l_tst)
# best_architecture_table(find_best_architecture(; s=:l_val, x=:i_tst, rexclude=[r"ILM"]); x=:i_tst)
# best_architecture_table(find_best_architecture(; s=:l_val, x=:r_tst, rexclude=[r"ILM"]); x=:r_tst)

# best_architecture_table(find_best_architecture(; s=:l_val, x=:l_tst, rinclude=[r"n=2"]); x=:l_tst)
# best_architecture_table(find_best_architecture(; s=:i_val, x=:i_tst, rinclude=[r"n=2"]); x=:i_tst)
# best_architecture_table(find_best_architecture(; s=:r_val, x=:r_tst, rinclude=[r"n=2"]); x=:r_tst)
