#!/usr/bin/env sh
#SBATCH --array=1-120
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1
#SBATCH --exclude=n33
#SBATCH --partition=cpu
#SBATCH --out=/home/rektomar/logs/spsn_clf_mip/%x-%j.out
#= 
ml --ignore_cache Julia/1.8.0-linux-x86_64
srun julia scripts/spsn_clf_mip.jl --n $SLURM_ARRAY_TASK_ID --m $1
exit    
# =#


"""
Experiment with fixed data seed, fixed model seed.
Using k seeds for data train/val/test spliting -> k-fold cross validation.
One seed for model initialization.
"""

const MODEL_SEED = 1;

using DrWatson
@quickactivate

using SumProductSet
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
using PoissonRandom
using LinearAlgebra
using EvalMetrics


const maxseed = 20
const path = "/home/$(ENV["USER"])/datasets/clean/mill"


function update_history!(history::NamedTuple, status) 
    for (key, value) in pairs(status)
        push!(history[key], value)
    end
    history
end

function train!(m::SumNode{T, <:SetNode}, x_trn::Mill.BagNode, y_trn::Vector{Int}; cb=iter->(), niter::Int=200, tol::Real=1e-5, opt=ADAM(0.02)) where T

    l_old = -Float64(Inf)
    ps = Flux.params(m)

    # compute gradient once to precompile computation
    gs = gradient(()-> sl_loss(m, x_trn, y_trn), ps);
    end_type = :mi  # max iters

    @info "Starting training"
    status = cb(0)

    start_time = time()
    for iter in 1:niter
        gs = gradient(()-> sl_loss(m, x_trn, y_trn), ps)            
        Flux.Optimise.update!(opt, ps, gs)

        status = cb(iter)

        l_trn = status[:l_trn]
        l_dif = l_trn - l_old
        l_old = l_trn

        if l_dif < 0 
            @info "STOPPED after $(iter) steps, obtained negative likelihood increment"
            end_type = :ni  # negative increment
            break
        end
        if abs(l_dif) < tol
            @info "STOPPED after $(iter) steps, reached minimum improvement tolerance"
            end_type = :tol  # tolerance
            break
        end
        if time() - start_time > 23.5*60*60
            @info "STOPPED after 23.5 hours of training"
            end_type = :time  # time_limit
            break
        end
    end

    @info "Finished training"
    trn_time = time() - start_time

    return trn_time, status
end

acc(prediction, target) = mean(prediction .== target)    

function status!(m, x_trn, x_val, x_tst, y_trn, y_val, y_tst, iter)
    l_trn = mean(logpdf(m, x_trn))
    l_val = mean(logpdf(m, x_val))
    l_tst = mean(logpdf(m, x_tst))

    ŷ_trn = getindex.(argmax(softmax(logjnt(m, x_trn)), dims=1), 1)[:]
    ŷ_val = getindex.(argmax(softmax(logjnt(m, x_val)), dims=1), 1)[:]
    ŷ_tst = getindex.(argmax(softmax(logjnt(m, x_tst)), dims=1), 1)[:]

    ari_trn = randindex(y_trn, ŷ_trn)[1]
    ari_val = randindex(y_val, ŷ_val)[1]
    ari_tst = randindex(y_tst, ŷ_tst)[1]
    ri_trn = randindex(y_trn, ŷ_trn)[2]
    ri_val = randindex(y_val, ŷ_val)[2]
    ri_tst = randindex(y_tst, ŷ_tst)[2]

    acc_trn = acc(ŷ_trn, y_trn)
    acc_val = acc(ŷ_val, y_val)
    acc_tst = acc(ŷ_tst, y_tst)

    # calculate scores
    s_trn = softmax(logjnt(m, x_trn), dims=1)
    s_val = softmax(logjnt(m, x_val), dims=1)
    s_tst = softmax(logjnt(m, x_tst), dims=1)

    # convert labels [1, 2] to [0, 1]
    auc_trn = binary_eval_report(y_trn .- 1, s_trn[2, :])["au_roccurve"]
    auc_val = binary_eval_report(y_val .- 1, s_val[2, :])["au_roccurve"]
    auc_tst = binary_eval_report(y_tst .- 1, s_tst[2, :])["au_roccurve"]

    @printf("Epoch %i - lkl:| %2.4e %2.4e %2.4e |  ari:| %.2f %.2f %.2f |  acc:| %.2f %.2f %.2f |   auc:| %.2f %.2f %.2f | \n",
        iter, l_trn, l_val, l_tst, ri_trn, ri_val, ri_tst, acc_trn, acc_val, acc_tst, auc_trn, auc_val, auc_tst)

    (; l_trn, l_val, l_tst, ari_trn, ari_val, ari_tst, ri_trn, ri_val, ri_tst, acc_trn, acc_val, acc_tst, auc_trn, auc_val, auc_tst, s_trn, s_val, s_tst, iter)
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
    # n - idx of cofing in grid
    # m - idx of dataset
    @add_arg_table s begin
        ("--n"; arg_type = Int; default=16);
        ("--m"; arg_type = Int; default=1);
    end
    parse_args(s)
end
function experiment(id::Int, dirdata::String, dataset_attributes::NamedTuple, grid)
    (; dataset, ndims, ndata, nbags) = dataset_attributes

    mtype = :setclassifier
    nb, ni, covtype, cardtype, nepoc, split, seed = collect(grid)[id]
    Random.seed!(seed)

    (; dirdata, dataset, ndims, ndata, nbags, split, seed, nb, ni, cardtype, nepoc, mtype, covtype, itype=Int64, ftype=Float64, ngrid=length(grid))
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
    i = mapreduce(seed->randperm(nbags), hcat, 1:maxseed)

    return ntuple2dict((; dataset, ndims, ndata, x, y, b, i))
end
function load_real_data(config::NamedTuple)
    (; ftype, split, seed) = config
    file, _ = produce_or_load(datadir("analysis_mip/datasets"),
                              config,
                              generate_real_data,
                              suffix="bson",
                              sort=false,
                              accesses=(:dataset, :ndims, :nbags, :ndata))
    @unpack x, y, b, i = file
    data = preprocess(ftype.(x), y, b, i[:, seed], ftype.(split))
    return data..., config
end

function estimate(config::NamedTuple)
    (; dataset, nb, ni, seed, covtype, nepoc, cardtype) = config
    x_trn, x_val, x_tst, y_trn, y_val, y_tst, config = load_real_data(config)

    @show dataset, nb, ni, covtype, cardtype, seed

    d = size(x_trn.data.data, 1)
    if cardtype == :poisson
        cdist = ()-> _Poisson()
    elseif cardtype == :categorical
        # kind of cheating
        k = maximum([length.(x_tst.bags); length.(x_val.bags); length.(x_trn.bags)])
        cdist = () -> _Categorical(k)
    else
        @error "Unknown cardinality distribution"
    end

    Random.seed!(MODEL_SEED)
    model = setmixture(nb, ni, d; cdist=cdist, Σtype=covtype)
    time, status = train!(model, x_trn, y_trn; cb=iter->status!(model, x_trn, x_val, x_tst, y_trn, y_val, y_tst, iter), niter=nepoc)
    n_params = sum(length, Flux.params(model))

    ntuple2dict(merge(config, status, (; model, n_params, time)))
end

# ranktable for old version of analysismip.jl
function result_table(; to_show=[:l, :ri, :ari, :acc], type=[:trn, :val, :tst], kwargs...)
    df = collect_results(datadir("analysis_mip/results"); kwargs...)

    table_metrics = [Symbol(metric, :_, settype) for metric in to_show for settype in type]
    # table_operations = [op => x->broadcast(last, x) for op in table_metrics]
    table_operations = [op => ByRow(x->last(x)) for op in table_metrics]
    df = combine(df, :dataset, :ni, :covtype, :cardtype, :seed, table_operations..., renamecols=false)
    df = groupby(df, [:dataset, :ni, :covtype, :cardtype])

    table_operations = [op => mean for op in table_metrics]

    df = combine(df, table_operations..., renamecols=false)
    df = combine(df->df[argmax(df[!, :l_val]), :], groupby(df, [:dataset]))
    combine(df, :dataset, :ni, :covtype, :cardtype, [xi => ByRow(n->round(n, sigdigits=3)) for xi in table_metrics]..., renamecols=false)
end

function main_local()
    Random.seed!(1)

    @load "$(path)/brown_creeper.bson" data labs bags
    perm = randperm(maximum(bags))
    x_trn, x_val, x_tst, y_trn, y_val, y_tst = preprocess(Float64.(data), labs, bags, perm)

    d = size(x_trn.data.data, 1)  # dimension of one instance
    nb = 2  # number of setmixture components
    ni = 8  # number of instance componenets

    mb_1 = setmixture(nb, ni, d)
    # mb_2 = setmixture(nb, 1, d; cdist=() -> _Categorical(50))

    niter = 10

    train!(mb_1, x_trn, y_trn; niter=niter, cb=iter->status!(mb_1, x_trn, x_val, x_tst, y_trn, y_val, y_tst, iter))
    # train!(mb_2, x_trn, y_trn; niter=niter, cb=iter->status!(mb_2, x_trn, x_val, x_tst, y_trn, y_val, y_tst, iter))

    nothing
end

function main_slurm()
    @unpack n, m = command_line()
    dataset = datasets[m]
    grid = Iterators.product(
        [2],
        [1 2 4 8 16 32],
        [:full, :diag],
        [:poisson, :categorical],
        [20000],
        [[64e-2, 16e-2, 2e-1]],
        collect(1:5))

        # nb, ni, covtype, cardtype, nepochs, train/val/test split, seeds
        # |grid| = 6 * 2 * 2 * 5 = 120

    produce_or_load(datadir("analysis_mip/results"),
                    experiment(n, dirdata, dataset, grid),
                    estimate;
                    suffix="jld2",
                    sort=false,
                    ignores=(:dirdata, :ngrid),
                    verbose=false,
                    force=false)
end

# main_local()
# main_slurm()
