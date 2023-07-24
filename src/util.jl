
const NODES = Union{SetNode, ProductNode, SumNode}

logpdf(m::NODES, x::AbstractVector{<:Mill.AbstractMillNode}) = Flux.ChainRulesCore.ignore_derivatives() do
    reduce(Mill.catobs, x)
end |> xr -> logpdf(m, xr)

Mill.getindex(x::Mill.ProductNode{<:NamedTuple}, k::Vector{Symbol})= Mill.ProductNode(x.data[k])

Base.hcat(A::SparseMatrixCSC...) = SparseArrays.sparse_hcat(A...)
Base.vcat(A::SparseMatrixCSC...) = SparseArrays.sparse_vcat(A...)
