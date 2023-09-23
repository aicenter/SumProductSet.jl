#!/usr/bin/env sh
#SBATCH --array=1-135
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1
#SBATCH --partition=cpulong
#SBATCH --exclude=n33
#SBATCH --out=/home/rektomar/logs/relational/%x-%j.out
#=
srun julia relational.jl --n $SLURM_ARRAY_TASK_ID --m $1
exit
# =#
using DrWatson
@quickactivate
using ArgParse
using DataFrames
using SumProductSet

include("utils.jl")


function commands()
    s = ArgParseSettings()
    @add_arg_table s begin
        ("--n"; arg_type = Int; default=1);
        ("--m"; arg_type = Int; default=1);
    end
    parse_args(s)
end

function slurm_spsn_acc()
    @unpack n, m = commands()
    dataset = datasets[m]
    pl, ps, nepoc, bsize, ssize, seed_split, seed_init = collect(Iterators.product(
        [2],
        collect(2:10),
        [200],
        [10],
        [1e-1, 1e-2, 1e-3],
        [1],
        collect(1:5)))[n]
    data = read("$(dirdata)/$(dataset.name).json", String)
    data = JSON3.read(data)
    x, y = data.x, data.y

    # x = reduce(catobs, suggestextractor(schema(x), (; scalar_extractors = default_scalar_extractor())).(x))
    x = reduce(catobs, suggestextractor(schema(x)).(x))
    x_trn, x_val, x_tst, y_trn, y_val, y_tst = split_data(x, y, seed_split)

    m = SumProductSet.reflectinmodel(x_trn[1], length(unique(y)); hete_nl=pl, hete_ns=ps, seed=seed_init)

    config_exp = (; seed_split, seed_init, dirdata, dataset=dataset.name, pl, ps, nepoc, bsize, ssize)
    config_wat = (suffix="jld2", sort=false, ignores=(:dirdata, ), verbose=true, force=true)

    gd!(m, x_trn, x_val, x_tst, y_trn, y_val, y_tst, Flux.Adam(ssize), nepoc, bsize, config_exp, config_wat, "accuracy")
end
function slurm_spsn_mis()
    @unpack n, m = commands()
    dataset = datasets[m]

    df = collect_results("data/nodewise_spsn/accuracy"; rinclude=[Regex(dataset.name)])
    df = groupby(df, [:dataset, :pl, :ps, :nepoc, :bsize, :ssize])
    df = combine(df, :acc_val=>mean, :acc_tst=>mean, renamecols=false)
	df = combine(df->df[argmax(df[!, :acc_val]), :], groupby(df, :dataset))

    mrate, seed_split, seed_init = collect(Iterators.product(
        [1e-4],
        [1],
        collect(1:10)))[n]
    @unpack pl, ps, nepoc, bsize, ssize = copy(df[1, :])

    data = read("$(dirdata)/$(dataset.name).json", String)
    data = JSON3.read(data)
    x, y = data.x, data.y

    x = reduce(catobs, suggestextractor(schema(x), (; scalar_extractors = default_scalar_extractor())).(x))
    x = make_missing(x, mrate)
    x_trn, x_val, x_tst, y_trn, y_val, y_tst = split_data(x, y, seed_split)

    m = SumProductSet.reflectinmodel(x_trn[1], length(unique(y)); hete_nl=pl, hete_ns=ps, seed=seed_init)

    config_exp = (; seed_split, seed_init, dirdata, dataset=dataset.name, pl, ps, nepoc, bsize, ssize, mrate)
    config_wat = (suffix="jld2", sort=false, ignores=(:dirdata, ), verbose=true, force=true)

    gd!(m, x_trn, x_val, x_tst, y_trn, y_val, y_tst, Flux.Adam(ssize), nepoc, bsize, config_exp, config_wat, "missing")
end
function slurm_spsn_ad()
    @unpack n, m = commands()
    dataset = datasets[m]
    seed_split, seed_init, pl, ps, nepoc, bsize, ssize = collect(Iterators.product(
        collect(1:5),
        [1],
        [2],
        collect(2:7),
        [200],
        [20],
        [1e-1, 1e-2, 1e-3]))[n]
    data = read("$(dirdata)/$(dataset.name).json", String)
    data = JSON3.read(data)
    x, y = data.x, data.y

    # x = reduce(catobs, suggestextractor(schema(x), (; scalar_extractors = default_scalar_extractor())).(x))
    x = reduce(catobs, suggestextractor(schema(x)).(x))
    x_trn, x_val, x_tst, _, y_val, y_tst = split_data_ad(x, y, seed_split)
    @show Mill.nobs(x_trn)
    @show Mill.nobs(x_val)
    @show Mill.nobs(x_tst)
    m = SumProductSet.reflectinmodel(x_trn[1], 1; hete_nl=pl, hete_ns=ps, seed=seed_init)

    config_exp = (; seed_split, seed_init, dirdata, dataset=dataset.name, pl, ps, nepoc, bsize, ssize)
    config_wat = (suffix="jld2", sort=false, ignores=(:dirdata, ), verbose=true, force=true)

    gd_ad!(m, x_trn, x_val, x_tst, y_val, y_tst, Flux.Adam(ssize), nepoc, bsize, config_exp, config_wat, "ad")
    @show rank(m, x_tst)
end

# slurm_spsn_acc()
# slurm_spsn_mis()
# slurm_spsn_ad()

nothing
