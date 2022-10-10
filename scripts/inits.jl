#!/usr/bin/env sh
#SBATCH --array=1-18
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1
#SBATCH --partition=cpu
#SBATCH --out=/home/rektomar/logs/%x-%j.out
#=
srun julia scripts/inits.jl --n $SLURM_ARRAY_TASK_ID 
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
const path = "/home/$(ENV["USER"])/datasets/clean/toy_pp"
const savefolder = "toy_pp_inits"
const dataset = "iidcluster"


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

    # y_trn = map(y->maximum(y), y_trn)
    # y_val = map(y->maximum(y), y_val)
    # y_tst = map(y->maximum(y), y_tst)

    ŷ_trn = getindex.(argmax(softmax(logjnt(m, x_trn)), dims=1), 1)
    ŷ_val = getindex.(argmax(softmax(logjnt(m, x_val)), dims=1), 1)
    ŷ_tst = getindex.(argmax(softmax(logjnt(m, x_tst)), dims=1), 1)

    i_trn = randindex(y_trn, ŷ_trn)[1]
    i_val = randindex(y_val, ŷ_val)[1]
    i_tst = randindex(y_tst, ŷ_tst)[1]


    @printf("lkl:| %2.4e %2.4e %2.4e |    ari:| %.2f %.2f %.2f |\n",
        l_trn, l_val, l_tst, i_trn, i_val, i_tst)

    (; l_trn, l_val, l_tst, i_trn, i_val, i_tst)
end

function command_line()
    s = ArgParseSettings()
    @add_arg_table s begin
        ("--n"; arg_type = Int; default=2);
    end
    parse_args(s)
end
function experiment(taskid::Int, dirdata::String, dataset::String, grid)

    mtype = :setmixture 
    nb, ni, μinit, Σinit, Σtype, nepoc, split, seed = collect(grid)[taskid]
    Random.seed!(seed)

    (; dirdata, dataset, split, seed, nb, ni, nepoc, μinit, Σinit, Σtype, mtype, itype=Int64, ftype=Float64, ngrid=length(grid))
end
function preprocess(x::AbstractArray{Tr,2}, y::AbstractArray{Ti,1}, b::AbstractArray,
    i::AbstractArray{Ti,1}=collect(1:length(b)), split::AbstractArray{Tr,1}=Tr.([64e-2, 16e-2, 2e-1])) where {Tr<:Real,Ti<:Int}
    
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

    y_trn = y[i[1:n[1]]]
    y_val = y[i[n[1]+1:n[2]]]
    y_tst = y[i[n[2]+1:end]]

    return x_trn, x_val, x_tst, y_trn, y_val, y_tst
end
function generate_real_data(config::NamedTuple)
    (; dirdata, dataset, ftype, itype, seed) = config

    @load "/home/$(ENV["USER"])/datasets/clean/$(dirdata)/$(dataset)_1.bson" data labs bags

    x = Matrix{ftype}(data)
    y = Vector{itype}(labs)
    b = Vector(bags) # unitranges
    nbags = length(b)
    i = mapreduce(seed->randperm(nbags), hcat, 1:maxseed)
    ndims, ndata = size(x)

    return ntuple2dict((; dataset, ndims, ndata, x, y, b, i))
end
function load_real_data(config::NamedTuple)
    (; ftype, split, seed) = config
    file, _ = produce_or_load(datadir("$(savefolder)/datasets"),
                              config,
                              generate_real_data,
                              suffix="bson",
                              sort=false,
                              accesses=(:dataset, :seed))
    @unpack x, y, b, i = file
    # fixed seed for preprocessing only !!!
    data = preprocess(ftype.(x), y, b, i[:, 1], ftype.(split))
    return data..., config
end
function estimate(config::NamedTuple)
    (; dataset, seed, nb, ni, nepoc, μinit, Σinit, Σtype, ftype) = config
    x_trn, x_val, x_tst, y_trn, y_val, y_tst, config = load_real_data(config)

    dtype = ftype
    ps = (; μinit, Σinit, Σtype, dtype)
    @show dataset, nb, ni, μinit, Σinit, Σtype, dtype, seed
    d = size(x_trn.data.data, 1)
    
    model = setmixture(nb, ni, d; ps...)
    status = train!(model, x_trn, x_val, x_tst, y_trn, y_val, y_tst; niter=nepoc)

    ntuple2dict(merge(config, status, (; model)))
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
    df = collect_results(datadir("$(resultfolder)/results"); kwargs...)
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
    @load "$(path)/iidcluster_1.bson" data labs bags
    x_trn, x_val, x_tst, y_trn, y_val, y_tst = preprocess(Float64.(data), labs, bags)

end

function main_slurm_real()
    @unpack n = command_line()
    grid = Iterators.product(
        [3],
        [1, 2],
        [:uniform, :randn],
        [:unit, :randlow],
        [:full, :diag],
        [100],
        [[64e-2, 16e-2, 2e-1]],
        collect(1:20))

        # nb, ni, μinit, Σinit, Σtype, nepochs, train/val/test split, seeds
        # |grid| = 2 * 2 * 2 * 2 * 20 = 320 < max_jobs = 400
        # TO DO: add learning rate to grid

    produce_or_load(datadir("$(savefolder)/results"),
                    experiment(n, "toy_pp", "iidcluster", grid),
                    estimate;
                    suffix="jld2",
                    sort=false,
                    ignores=(:dirdata, :ngrid),
                    verbose=false)
end

# main_local_real -> experiment config -> estimate -> load_real_data -> generate_real_data

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
