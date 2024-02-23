using Test
using SumProductSet, Flux, Distributions, SparseArrays, ChainRulesCore, ChainRulesTestUtils
import Mill

include("distributions/mvnormal.jl")
include("distributions/poisson.jl")
include("distributions/categorical.jl")
include("distributions/geometric.jl")


include("setnode.jl")
include("productnode.jl")
# include("modelbuilders.jl")
