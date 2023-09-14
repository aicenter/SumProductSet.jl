using Mill
using JSON3
using Random
using JsonGrinder
using StatsBase: sample


dirdata = "/home/$(ENV["USER"])/datasets/clean/relational"
# datasets = [
#     (name="mutagenesis", ndata=188,   nclass=2 ) # 1
#     (name="genes",       ndata=862,   nclass=15) # 2
#     (name="cora",        ndata=2708,  nclass=7 ) # 3
#     (name="citeseer",    ndata=3312,  nclass=6 ) # 4
#     (name="webkp",       ndata=877,   nclass=5 ) # 5
#     (name="world",       ndata=239,   nclass=7 ) # 6
#     (name="chess",       ndata=295,   nclass=3 ) # 7
#     (name="uw_cse",      ndata=278,   nclass=4 ) # 8
#     (name="hepatitis",   ndata=500,   nclass=2 ) # 9
#     # (name="ftp",         ndata=30000, nclass=3 ) # 10
#     # (name="craft_beer",  ndata=558,   nclass=51) # 11
# ]
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
    (name="pubmed_diabetes", ndata=19717, nclass=3 ) # 11
    (name="ftp",             ndata=30000, nclass=3 ) # 12
    (name="ptc",             ndata=343,   nclass=2 ) # 13
    (name="dallas",          ndata=219,   nclass=7 ) # 14
    (name="premier_league",  ndata=380,   nclass=3 ) # 15
]

attributes = (dataset=map(d->d.name, datasets), data_per_class=map(d->round(Int, d.ndata/d.nclass), datasets))


function split_data(x::Mill.AbstractMillNode, y::AbstractArray{Ti,1}, seed::Ti=Ti(1), ratio::Array{Tr,1}=[64f-2, 16f-2, 2f-1]) where {Tr<:Real,Ti<:Int}
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

function split_data_ad(x::Mill.AbstractMillNode, y::AbstractArray{Ti,1}, seed::Ti=Ti(1), ratio::Array{Tr,1}=[64f-2, 16f-2, 2f-1]) where {Tr<:Real,Ti<:Int}
    Random.seed!(seed)
    # shuffle data
    i = randperm(length(y))
    x = x[i]
    y = y[i]

    nclass = length(unique(y))
    # assume that seed is mostly ∈ <0, nclass>
    in_class = seed > nclass ? rand(1:nclass) : seed
    in_mask = y.==in_class

    x_in = x[in_mask]
    x_out =x[.!in_mask]
    
    n_in = cumsum(map(n->round(Ti, n), ratio*length(x_in)))  # changed ceil -> round

    l_out = length(x_out)
    s_out = length(x_in)-n_in[1]
    n_out = n_in[2:3] .- n_in[1]
    if l_out < s_out
        r = s_out/l_out
        s_out = l_out 
        
        n_out .= floor.(Int, r.*n_out)
    end

    i_out = sample(1:l_out, s_out; replace=false) # for val and tst only

    x_out = x_out[i_out]

    # in distribution val/tst
    x_in_val = x_in[n_in[1]+1:n_in[2]]
    x_in_tst = x_in[n_in[2]+1:end]

    # out of distribution val/tst

    x_out_val = x_out[1:n_out[1]]
    x_out_tst = x_out[n_out[1]+1:end]

    # final trn/val/tst split
    x_trn = x_in[1:n_in[1]]

    x_val = Mill.catobs(x_in_val, x_out_val)
    @show Mill.nobs(x_in_val), Mill.nobs(x_out_val)
    y_val = vcat(zeros(Ti, length(x_in_val)), ones(Ti, length(x_out_val)))

    x_tst = Mill.catobs(x_in_tst, x_out_tst)
    @show Mill.nobs(x_in_tst), Mill.nobs(x_out_tst)
    y_tst = vcat(zeros(Ti, length(x_in_tst)), ones(Ti, length(x_out_tst)))

    x_trn, x_val, x_tst, nothing, y_val, y_tst
end


function default_scalar_extractor()
	[
	(e -> length(keys(e)) <= 100 && JsonGrinder.is_numeric_or_numeric_string(e),
		(e, uniontypes) -> ExtractCategorical(keys(e), uniontypes)),
	(e -> JsonGrinder.is_intable(e),
		(e, uniontypes) -> JsonGrinder.extractscalar(Int32, e, uniontypes)),
	(e -> JsonGrinder.is_floatable(e),
	 	(e, uniontypes) -> JsonGrinder.extractscalar(FloatType, e, uniontypes)),
	(e -> (keys_len = length(keys(e)); keys_len < 1000 && !JsonGrinder.is_numeric_or_numeric_string(e)),
		(e, uniontypes) -> ExtractCategorical(keys(e), uniontypes)),
	(e -> true,
		(e, uniontypes) -> JsonGrinder.extractscalar(JsonGrinder.unify_types(e), e, uniontypes)),]
end
