#!/usr/bin/env sh
#SBATCH --array=1-135
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1
#SBATCH --partition=cpu
#SBATCH --exclude=n33
#SBATCH --out=/home/papezmil/logs/%x-%j.out
#=
srun julia nodewise_spsn.jl --n $SLURM_ARRAY_TASK_ID --m $1
exit
# =#
using DrWatson
@quickactivate
using Flux
using JSON3
using Random
using Printf
using ArgParse
using Statistics
using JsonGrinder
using SumProductSet
using HierarchicalUtils
import Mill


function logjnt(m::SumNode, x::Union{AbstractMatrix, Mill.AbstractMillNode}, y)
    l = mapreduce(c->logpdf(c, x), vcat, m.components) .+ logsoftmax(m.weights)
    l[CartesianIndex.(y, 1:length(y))]
end


Base.length(x::Mill.ProductNode) = Mill.nobs(x)
predict(m, x) = mapslices(argmax, SumProductSet.logjnt(m, x), dims=1)[:]


function evaluate(m, x_trn::Ar, x_val::Ar, x_tst::Ar, y_trn::Ai, y_val::Ai, y_tst::Ai) where {Ar<:Mill.AbstractMillNode,Ai<:AbstractArray{<:Int,1}}
    lkl_trn = mean(logpdf(m, x_trn))
    lkl_val = mean(logpdf(m, x_val))
    lkl_tst = mean(logpdf(m, x_tst))

    acc_trn = mean(y_trn .== predict(m, x_trn))
    acc_val = mean(y_val .== predict(m, x_val))
    acc_tst = mean(y_tst .== predict(m, x_tst))

    (; lkl_trn, lkl_val, lkl_tst, acc_trn, acc_val, acc_tst)
end

function gd!(m, x_trn::Ar, x_val::Ar, x_tst::Ar,
                y_trn::Ai, y_val::Ai, y_tst::Ai,
                opt, nepoc::Int, bsize::Int, supervision::Tr=0f0; ps::Flux.Params=Flux.params(m), eps_abs::Tr=Tr(-1f-16)) where {Ar<:Mill.AbstractMillNode,Ai<:AbstractArray{<:Int,1},Tr<:Real}
    t_trn = Tr[]
    l_trn, l_val, l_tst = Tr[], Tr[], Tr[]
    a_trn, a_val, a_tst = Tr[], Tr[], Tr[]
    l_old = -Tr(Inf)
    final = :maximum_iterations
    # s = round(Int, supervision*length(x_trn))

    d_trn = Flux.DataLoader((x_trn, y_trn); batchsize=bsize)

    for e in 1:nepoc
        t̄_trn = @elapsed begin
            for (x_trn, y_trn) in d_trn
                if     supervision == Tr(1f0)
                    gs = gradient(()->-mean(logjnt(m, x_trn, y_trn)), ps)
                elseif supervision == Tr(0f0)
                    gs = gradient(()->-mean(logpdf(m, x_trn)), ps)
                else
                    # gs = gradient(()->-mean(logjnt(m, x_trn[:, 1:s], y_trn[1:s]))-mean(logpdf(m, x_trn[:, s+1:end])), ps)
                end
                Flux.Optimise.update!(opt, ps, gs)
            end
        end

        l̄_trn, l̄_val, l̄_tst, ā_trn, ā_val, ā_tst = evaluate(m, x_trn, x_val, x_tst, y_trn, y_val, y_tst)

        l_dif = l̄_trn - l_old
        l_old = l̄_trn

        if abs(l_dif) <= eps_abs
            final = :absolute_tolerance
            break
        end
        if isnan(l̄_trn)
            final = :nan
            break
        end

        if mod(e, 10) == 1
            push!(t_trn, t̄_trn)
            push!(l_trn, l̄_trn)
            push!(l_val, l̄_val)
            push!(l_tst, l̄_tst)
            push!(a_trn, ā_trn)
            push!(a_val, ā_val)
            push!(a_tst, ā_tst)
        end

        @printf("gd: epoch: %i | l_trn %2.2f | l_val %2.2f | l_tst %2.2f || a_trn %2.2f | a_val %2.2f | a_tst %2.2f |\n", e, l̄_trn, l̄_val, l̄_tst, ā_trn, ā_val, ā_tst)
    end

    println("status: $(final)")

    (; t_trn, l_trn, l_val, l_tst, final)
end




function split(x::Mill.AbstractMillNode, y::AbstractArray{Ti,1}, seed::Ti=Ti(1), ratio::Array{Tr,1}=[64f-2, 16f-2, 2f-1]) where {Tr<:Real,Ti<:Int}
    Random.seed!(seed)
    i = randperm(length(y))
    n = cumsum(map(n->ceil(Ti, n), ratio*length(x)))

    x_trn = x[i[1:n[1]]]
    x_val = x[i[n[1]+1:n[2]]]
    x_tst = x[i[n[2]+1:end]]

    y_trn = y[i[1:n[1]]]
    y_val = y[i[n[1]+1:n[2]]]
    y_tst = y[i[n[2]+1:end]]

    x_trn, x_val, x_tst, y_trn, y_val, y_tst
end


dirdata = "/home/papezmil/datasets/clean/relational"
datasets = [
    (name="mutagenesis",     ndata=188,   nclass=2 ) # 1
    (name="genes",           ndata=862,   nclass=15) # 2
    (name="cora",            ndata=2708,  nclass=7 ) # 3
    (name="citeseer",        ndata=3312,  nclass=6 ) # 4
    (name="webkp",           ndata=877,   nclass=5 ) # 5
    (name="world",           ndata=239,   nclass=7 ) # 6
    (name="craft_beer",      ndata=558,   nclass=51) # 7
    (name="chess",           ndata=295,   nclass=3 ) # 8
    (name="uw_cse",          ndata=278,   nclass=4 ) # 9
    (name="hepatitis",       ndata=500,   nclass=2 ) # 10
]

function commands()
    s = ArgParseSettings()
    @add_arg_table s begin
        ("--n"; arg_type = Int; default=1);
        ("--m"; arg_type = Int; default=1);
    end
    parse_args(s)
end
function estimate(config::NamedTuple)
    (; dirdata, dataset, seed_split, seed_init, pl, ps, nepoc, bsize, supervision, msave) = config

	data = read("$(dirdata)/$(dataset).json", String)
	data = JSON3.read(data)
    x, y = data.x, data.y

    s = JsonGrinder.schema(x)
    e = suggestextractor(s)
    x = Mill.catobs(e.(x))
    x_trn, x_val, x_tst, y_trn, y_val, y_tst = split(x, y, seed_split)

    Random.seed!(seed_init)

    m = reflectinmodel(x_trn[1], length(unique(y)); hete_nl=pl, hete_ns=ps)

    record = gd!(m, x_trn, x_val, x_tst, y_trn, y_val, y_tst, Adam(), nepoc, bsize, supervision)

    if msave == true
        ntuple2dict(merge(config, evaluate(m, x_trn, x_val, x_tst, y_trn, y_val, y_tst), record, (; m)))
    else
        ntuple2dict(merge(config, evaluate(m, x_trn, x_val, x_tst, y_trn, y_val, y_tst), record))
    end
end


function slurm()
    @unpack n, m = commands()
    dataset = datasets[m]
    pl, ps, nepoc, bsize, supervision, seed_split, seed_init = collect(Iterators.product(
        [2, 3],
        [2, 3],
        [400],
        [10, 20, 30],
        [1f-0],
        [1],
        collect(1:5)))[n]

    settings = (; seed_split, seed_init, dirdata, dataset=dataset.name, pl, ps, nepoc, bsize, supervision, rtype=Float32, itype=Int64, msave=false)

    result, _ = produce_or_load(datadir("relational"),
                    settings,
                    estimate;
                    suffix="jld2",
                    sort=false,
                    ignores=(:dirdata, ),
                    verbose=false)

    # result = estimate(settings)

    display(result)
end

slurm()

nothing
