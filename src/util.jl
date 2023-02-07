
NODES = Union{SetNode, ProductNode, SumNode}

logpdf(m::NODES, x::AbstractVector{<:Mill.AbstractMillNode}) = Flux.ChainRulesCore.ignore_derivatives() do
    reduce(Mill.catobs, x)
end |> xr -> logpdf(m, xr)
