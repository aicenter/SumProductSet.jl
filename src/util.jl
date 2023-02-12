
NODES = Union{SetNode, ProductNode, SumNode}

logpdf(m::NODES, x::AbstractVector{<:Mill.AbstractMillNode}) = Flux.ChainRulesCore.ignore_derivatives() do
    reduce(Mill.catobs, x)
end |> xr -> logpdf(m, xr)

Mill.getindex(x::Mill.ProductNode{<:NamedTuple}, k::Union{Vector{Symbol}, NTuple{N, Symbol}}) where N = Mill.ProductNode(x.data[k])
Mill.getindex(x::Mill.ProductNode{<:NamedTuple}, k::Set{Symbol}) = Mill.getindex(x, collect(k))
