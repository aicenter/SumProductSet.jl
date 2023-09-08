using Mill
using JSON3
using Random
using JsonGrinder


dirdata = "/home/$(ENV["USER"])/datasets/clean/relational"
datasets = [
    (name="mutagenesis", ndata=188,   nclass=2 ) # 1
    (name="genes",       ndata=862,   nclass=15) # 2
    (name="cora",        ndata=2708,  nclass=7 ) # 3
    (name="citeseer",    ndata=3312,  nclass=6 ) # 4
    (name="webkp",       ndata=877,   nclass=5 ) # 5
    (name="world",       ndata=239,   nclass=7 ) # 6
    (name="craft_beer",  ndata=558,   nclass=51) # 7
    (name="chess",       ndata=295,   nclass=3 ) # 8
    (name="uw_cse",      ndata=278,   nclass=4 ) # 9
    (name="hepatitis",   ndata=500,   nclass=2 ) # 10
    # (name="ftp",         ndata=30000, nclass=3 ) # 11
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
