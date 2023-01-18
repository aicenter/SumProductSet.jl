"""
    AbstractModelNode
Supertype for any model node defined in `SumProductSet.jl`.
"""
abstract type AbstractModelNode end

include("setnode.jl")
include("sumnode.jl")
include("productnode.jl")
