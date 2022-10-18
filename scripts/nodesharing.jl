#!/usr/bin/env sh
#SBATCH --array=1-48
#SBATCH --mem=16G
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1
#SBATCH --partition=cpu
#SBATCH --out=/home/rektomar/logs/nodesharing/%x-%j.out
#=
srun julia scripts/nodesharing.jl --n $SLURM_ARRAY_TASK_ID --m $1
exit
# =#

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
using SpecialFunctions


const maxseed = 20
const path = "/home/$(ENV["USER"])/datasets/clean/mill"


function train!(m::SumNode{T, <:SetNode}, x_trn::S, x_val::S, x_tst::S, y_trn, y_val, y_tst; niter::Int=100, opt=ADAM(0.01)) where {T, S<:Mill.BagNode}
    # @printf("model: %s:\n", m.name)

    ps = Flux.params(m)

    status = []

    for _ in 1:niter
        gs = gradient(()->-mean(logpdf(m, x_trn)), ps)
        Flux.Optimise.update!(opt, ps, gs)
        status = status!(m, x_trn, x_val, x_tst, y_trn, y_val, y_tst, true)
    end

    return status
end


function status!(m, x_trn, x_val, x_tst, y_trn, y_val, y_tst, verbose=false)
    l_trn = mean(logpdf(m, x_trn))
    l_val = mean(logpdf(m, x_val))
    l_tst = mean(logpdf(m, x_tst))

    y_trn = map(y->maximum(y), y_trn)
    y_val = map(y->maximum(y), y_val)
    y_tst = map(y->maximum(y), y_tst)

    ŷ_trn = getindex.(argmax(softmax(logjnt(m, x_trn)), dims=1), 1)
    ŷ_val = getindex.(argmax(softmax(logjnt(m, x_val)), dims=1), 1)
    ŷ_tst = getindex.(argmax(softmax(logjnt(m, x_tst)), dims=1), 1)

    # if isa(m, ILM1F)
    #     ŷ_trn = map(j->maximum(ŷ_trn[j]), x_trn.bags)
    #     ŷ_val = map(j->maximum(ŷ_val[j]), x_val.bags)
    #     ŷ_tst = map(j->maximum(ŷ_tst[j]), x_tst.bags)     
    # end

    h_trn = vmeasure(y_trn, ŷ_trn; β=1e-4)
    h_val = vmeasure(y_val, ŷ_val; β=1e-4)
    h_tst = vmeasure(y_tst, ŷ_tst; β=1e-4)
    c_trn = vmeasure(y_trn, ŷ_trn; β=0.5)
    c_val = vmeasure(y_val, ŷ_val; β=0.5)
    c_tst = vmeasure(y_tst, ŷ_tst; β=0.5)
    i_trn = randindex(y_trn, ŷ_trn)[1]
    i_val = randindex(y_val, ŷ_val)[1]
    i_tst = randindex(y_tst, ŷ_tst)[1]
    r_trn = randindex(y_trn, ŷ_trn)[2]
    r_val = randindex(y_val, ŷ_val)[2]
    r_tst = randindex(y_tst, ŷ_tst)[2]
    v_trn = Clustering.varinfo(y_trn, ŷ_trn)
    v_val = Clustering.varinfo(y_val, ŷ_val)
    v_tst = Clustering.varinfo(y_tst, ŷ_tst)

    # if m.n == 2
    # mean absolute error
    a_trn = sum(abs.(y_trn - vec(ŷ_trn))) / length(y_trn)
    a_val = sum(abs.(y_val - vec(ŷ_val))) / length(y_val)
    a_tst = sum(abs.(y_tst - vec(ŷ_tst))) / length(y_tst)

    if verbose == false
        @printf("lkl:| %2.4e %2.4e %2.4e |    ari:| %.2f %.2f %.2f |\n",
            l_trn, l_val, l_tst, i_trn, i_val, i_tst)
    else
        @printf("lkl:| %2.4e %2.4e %2.4e |    h:| %.2f %.2f %.2f |    c:| %.2f %.2f %.2f |    ari:| %.2f %.2f %.2f |    ri:| %.2f %.2f %.2f |    vi:| %.2f %.2f %.2f |    a:| %.2f %.2f %.2f | \n",
            l_trn, l_val, l_tst, h_trn, h_val, h_tst, c_trn, c_val, c_tst, i_trn, i_val, i_tst, r_trn, r_val, r_tst, v_trn, v_val, v_tst, a_trn, a_val, a_tst)
    end
    # else
    #     if verbose == false
    #         @printf("lkl:| %2.4e %2.4e %2.4e |    ari:| %.2f %.2f %.2f |\n",
    #             l_trn, l_val, l_tst, i_trn, i_val, i_tst)
    #     else
    #         @printf("lkl:| %2.4e %2.4e %2.4e |    h:| %.2f %.2f %.2f |    c:| %.2f %.2f %.2f |    ari:| %.2f %.2f %.2f |    ri:| %.2f %.2f %.2f |    vi:| %.2f %.2f %.2f | \n",
    #             l_trn, l_val, l_tst, h_trn, h_val, h_tst, c_trn, c_val, c_tst, i_trn, i_val, i_tst, r_trn, r_val, r_tst, v_trn, v_val, v_tst)
    #     end
    # end

    (; l_trn, l_val, l_tst, h_trn, h_val, h_tst, c_trn, c_val, c_tst, i_trn, i_val, i_tst, r_trn, r_val, r_tst, v_trn, v_val, v_tst)
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
        ("--n"; arg_type = Int; default=2);
        ("--m"; arg_type = Int; default=1);
    end
    parse_args(s)
end
function experiment(id::Int, dirdata::String, dataset_attributes::NamedTuple, grid)
    (; dataset, ndims, ndata, nbags) = dataset_attributes

    mtype = :setmixture # fixed for now
    nb, ni_n, ni_s, ctype, nepoc, split, seed = collect(grid)[id]
    Random.seed!(seed)

    (; dirdata, dataset, ndims, ndata, nbags, split, seed, nb, ni_n, ni_s, nepoc, mtype, ctype, itype=Int64, ftype=Float64, ngrid=length(grid))
end
function preprocess(x::AbstractArray{Tr,2}, y::AbstractArray{Ti,1}, b::AbstractArray{Ti,1}, i::AbstractArray{Ti,1}=collect(1:maximum(b)), split::AbstractArray{Tr,1}=Tr.([64e-2, 16e-2, 2e-1])) where {Tr<:Real,Ti<:Int}
    x = x[vec(std(x, dims=2) .> Tr(1e-5)), :]

    x = Mill.BagNode(x, b)
    n = cumsum(map(n->ceil(Ti, n), split*nobs(x)))

    x_trn = x[i[1:n[1]]]
    x_val = x[i[n[1]+1:n[2]]]
    x_tst = x[i[n[2]+1:end]]

    mn = mean(x_trn.data.data, dims=2)
    sd = std(x_trn.data.data, dims=2)
    f(z) = (z .- mn) ./ sd
    x_trn = Mill.mapdata(f, x_trn)
    x_val = Mill.mapdata(f, x_val) 
    x_tst = Mill.mapdata(f, x_tst)  

    y_trn = map(j->y[j], x.bags[i[1:n[1]]])
    y_val = map(j->y[j], x.bags[i[n[1]+1:n[2]]])
    y_tst = map(j->y[j], x.bags[i[n[2]+1:end]])

    return x_trn, x_val, x_tst, y_trn, y_val, y_tst
end
function generate_real_data(config::NamedTuple)
    (; dirdata, dataset, ndata, nbags, ftype, itype) = config

    @load "/home/$(ENV["USER"])/datasets/clean/$(dirdata)/$(dataset).bson" data labs bags

    x = Matrix{ftype}(data)
    y = Vector{itype}(labs)
    b = Vector{itype}(bags)
    i = mapreduce(seed->randperm(nbags), hcat, 1:maxseed)

    tostringdict(ntuple2dict((; dataset, ndims, ndata, x, y, b, i)))
end
function load_real_data(config::NamedTuple)
    (; ftype, split, seed) = config
    file, _ = produce_or_load(datadir("spsn_node_sharing/datasets"),
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
    (; dataset, nb, ni_n, ni_s, seed, mtype, ctype, nepoc, ftype) = config
    x_trn, x_val, x_tst, y_trn, y_val, y_tst, config = load_real_data(config)

    @show dataset, nb, (ni_n, ni_s), seed

    # model = getfield(@__MODULE__, mtype){ftype,ctype}(n, m, size(x_trn.data.data, 1))
    d = size(x_trn.data.data, 1)
    # model = setmixture(n, m, d)
    model = sharedsetmixture(nb, ni_s, ni_n, d)
    status = train!(model, x_trn, x_val, x_tst, y_trn, y_val, y_tst; niter=nepoc)

    tostringdict(ntuple2dict(merge(config, status, (; model))))
end


function rank_table(df::DataFrame, highlight::Array{<:NamedTuple,1}; latex::Bool=false)
    push!(df, ["rank", map(c->typeof(c)==String ? "" : 0f0, df[1, 2:end])...])
    foreach(highlight) do h
        df[end, h.vs] = mean(map(r->StatsBase.competerank(Vector(r[h.vs]), rev=h.rev), eachrow(df[1:end-1, :])))
    end

    if !latex
        h = map(highlight) do h
            h1 = Highlighter(f=(data, i, j)->i< size(df, 1)&&data[i,j]==h.fa(df[i, h.vs]), crayon=crayon"yellow bold")
            h2 = Highlighter(f=(data, i, j)->i==size(df, 1)&&data[i,j]==h.fb(df[i, h.vs]), crayon=crayon"yellow bold")
            h1, h2
        end
        pretty_table(df, formatters=ft_round(2), highlighters=(collect(Iterators.flatten(h))...,))
    else
        h = map(highlight) do h
            h1 = LatexHighlighter((data, i, j)->i< size(df, 1)&&data[i,j]==h.fa(df[i, h.vs]), ["color{blue}","textbf"])
            h2 = LatexHighlighter((data, i, j)->i==size(df, 1)&&data[i,j]==h.fb(df[i, h.vs]), ["color{blue}","textbf"])
            h1, h2
        end
        pretty_table(df, formatters=ft_round(2), backend=:latex, highlighters=(collect(Iterators.flatten(h))...,))
    end
end
function find_best_architecture(; s::Symbol=:l_val, x::Symbol=:l_tst, kwargs...)
    df = collect_results(datadir("spsn_node_sharing/results"); kwargs...)
    df = groupby(df, [:dataset, :m, :n, :mtype])
    df = combine(df, s=>mean, x=>mean, x=>std=>:std, renamecols=false)
    combine(df->df[argmax(df[!, s]), :], groupby(df, [:dataset, :m]))  # groupby(df, [:dataset, :mtype])
end
function best_architecture_table(df::DataFrame; x::Symbol=:l_tst)
    gf = groupby(df, :m)
    df = map(pairs(gf)) do (k, v)
        @show k
        @show v
        name = string(k[1])
        if contains(name, "2")
            combine(v, :dataset, x => ByRow(n->round(n, sigdigits=3))=>"$(x)_"*name,
                              :std => ByRow(n->round(n, sigdigits=3))=>"std_"*name,
                          [:n, :m] => ByRow((n, m)->"$(n)-$(m)")=>"n_m_"*name, renamecols=false)
        else
            combine(v, :dataset, x => ByRow(n->round(n, sigdigits=3))=>"$(x)_"*name,
                              :std => ByRow(n->round(n, sigdigits=3))=>"std_"*name,
                                :n => ByRow(n->"$(n)")=>"n_"*name, renamecols=false)
        end
    end
    # @showdf
    df = innerjoin(df...; on=:dataset, makeunique=true)

    ls = Symbol.(filter(name->contains(name, "$(x)"), names(df)))
    ss = Symbol.(filter(name->contains(name, "std" ), names(df)))
    ns = Symbol.(filter(name->contains(name, "n"   ), names(df)))
    ms = Symbol.(filter(name->contains(name, "m"   ), names(df)))

    df = combine(df, :dataset, ls, ss, ns, ms)
    rank_table(df, [(vs=ls, fa=maximum, fb=minimum, rev=true), (vs=ss, fa=minimum, fb=maximum, rev=true)]; latex=false)
end


function main_local_real()
    # Random.seed!(1)

    @load "$(path)/brown_creeper.bson" data labs bags
    x_trn, x_val, x_tst, y_trn, y_val, y_tst = preprocess(Float64.(data), labs, bags)

    d = size(x_trn.data.data, 1)
    n = 2

    mb_1f = setmixture(n, 1, d)
    mb_2f = setmixture(n, 3, d)

    niter = 100

    # train!(deepcopy(mi_1f), x_trn, x_val, x_tst, y_trn, y_val, y_tst; niter=niter)
    train!(deepcopy(mb_1f), x_trn, x_val, x_tst, y_trn, y_val, y_tst; niter=niter)
    train!(deepcopy(mb_2f), x_trn, x_val, x_tst, y_trn, y_val, y_tst; niter=niter)

    nothing
end

function main_slurm_real()
    @unpack n, m = command_line()
    dataset = datasets[m]
    grid = Iterators.product(
        [2],
        [1 2 4 8],
        [0 2 4 8],
        [:full],
        [500],
        [[64e-2, 16e-2, 2e-1]],
        collect(1:3))

        # nb, ni_n, ni_s, covtype, nepochs, train/val/test split, seeds
        # |grid| = 4 * 4 * 3 = 48 
        # n_dataset * |grid| = 20 * 18 = 360 < max_jobs = 400
        # TO DO: add learning rate to grid

    produce_or_load(datadir("spsn_node_sharing/results"),
                    experiment(n, dirdata, dataset, grid),
                    estimate;
                    suffix="jld2",
                    sort=false,
                    ignores=(:dirdata, :ngrid),
                    verbose=false)
end


# main_local_real()
# main_slurm_real()

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
