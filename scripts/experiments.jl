using DrWatson
@quickactivate
using Mill
using Flux
using JSON3
using Revise
using Statistics
using JsonGrinder
using SumProductSet


# (name="mutagenesis",     ndata=188,   nclass=2 ) # 1
# (name="genes",           ndata=862,   nclass=15) # 2
# (name="cora",            ndata=2708,  nclass=7 ) # 3
# (name="citeseer",        ndata=3312,  nclass=6 ) # 4
# (name="webkp",           ndata=877,   nclass=5 ) # 5
# (name="world",           ndata=239,   nclass=7 ) # 6
# (name="craft_beer",      ndata=558,   nclass=51) # 7
# (name="chess",           ndata=295,   nclass=3 ) # 8
# (name="uw_cse",          ndata=278,   nclass=4 ) # 9
# (name="hepatitis",       ndata=500,   nclass=2 ) # 10


data = read("/home/papezmil/datasets/clean/relational/mutagenesis.json", String)
data = JSON3.read(data)
x, y = data.x, data.y

s = JsonGrinder.schema(x)
e = suggestextractor(s)
x = Mill.catobs(e.(x))

m = SumProductSet.reflectinmodel(x[1], length(unique(y)); depth_prod=2, n_prod_mix=2)

# printtree(s, htrunc=25, vtrunc=25)
# printtree(x, htrunc=25, vtrunc=25)
printtree(m, htrunc=25, vtrunc=25)

logpdf(m, x)

ps = Flux.params(m)
gs = gradient(()->-mean(logpdf(m, x)), ps)
foreach(p->display(gs[p]), ps)
