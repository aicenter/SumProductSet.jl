
mutable struct ModelSettings
    root_ns::Integer
    homo_ns::Integer
    hete_nl::Integer
    hete_ns::Vector{<:Integer}
    dist_cont::Function
    dist_disc::Function
    dist_gram::Function
    dist_card::Function
    dist_bin::Function
    data_type::Type{<:Real}
end

function decrease_hete_ns!(m::ModelSettings)
    m.hete_ns = m.hete_ns[2:end]
    return m
end
"""
    reflectinmodel(x, root_ns; kwargs...)

Build mixture model of HMILL data.
x - a signle HMILL sample.
root_ns - a number of mixture components in the root node (for classification, it is equal to number of classes).

# Examples

```julia-repl
z = Mill.ArrayNode([0. 1; 2 3]);
x = Mill.ProductNode(a=z, b=z, c=z, d=z);
reflectinmodel(x, 2; hete_nl=2, hete_ns=2)
```
"""
function reflectinmodel(
        x::Mill.AbstractMillNode,
        root_ns::Integer;
        homo_ns::Int = 1,
        hete_nl::Int = 1,
        hete_ns::Int = 1,
        dist_cont = d->gmm(2, d),
        dist_disc = d->Categorical(d),
        dist_gram = d->Geometric(d),
        dist_card = ()->Poisson(),
        dist_bin = d->MvBernoulli(d),
        data_type::Type{<:Real} = Float32,
        seed::Int=1
    )

    settings = ModelSettings(root_ns, homo_ns, hete_nl, repeat([hete_ns], 9999), dist_cont, dist_disc, dist_gram, dist_card, dist_bin, data_type)

    Random.seed!(seed)

    root_ns > 1 ? SumNode(map(_->_reflectinmodel(x, settings), 1:root_ns)) : _reflectinmodel(x, settings) 
end

function _reflectinmodel(x::Mill.ProductNode, settings::ModelSettings)
    if settings.hete_nl == 1
        _productmodel(x,                                     settings.hete_ns[1], decrease_hete_ns!(settings))
    else
        _productmodel(x, collect(keys(x)), settings.hete_nl, settings.hete_ns[1], decrease_hete_ns!(settings))
    end
end
function _reflectinmodel(x::Mill.BagNode,     settings::ModelSettings)
    if settings.homo_ns == 1
        SetNode(_reflectinmodel(x.data, settings), settings.dist_card())
    else
        SumNode(map(_->SetNode(_reflectinmodel(x.data, settings), settings.dist_card()), 1:settings.homo_ns))
    end
end
_reflectinmodel(x::Mill.ArrayNode, settings::ModelSettings) = _reflectinmodel(x.data, settings)

_reflectinmodel(x::OneHotArray,           settings)                     = settings.dist_disc(size(x, 1))
_reflectinmodel(x::MaybeHotArray,         settings)                     = settings.dist_disc(size(x, 1))
_reflectinmodel(x::Array{T},              settings) where T <: Real     = settings.dist_cont(size(x, 1))
_reflectinmodel(x::Array{Maybe{T}},       settings) where T <: Real     = settings.dist_cont(size(x, 1))
_reflectinmodel(x::NGramMatrix{T},        settings) where T <: Sequence = settings.dist_gram(size(x, 1))
_reflectinmodel(x::NGramMatrix{Maybe{T}}, settings) where T <: Sequence = settings.dist_gram(size(x, 1))
_reflectinmodel(x::BitMatrix,             settings)                     = settings.dist_bin(size(x, 1))


function _productmodel(x, n::Int, settings::ModelSettings)
    k = keys(x.data)
    c = map(_->ProductNode(mapreduce(k->_reflectinmodel(x.data[k], settings), vcat, k), reduce(vcat, k)), 1:n)
    n == 1 ? first(c) : SumNode(c)
end
function _productmodel(x, scope::Vector{Symbol}, l::Int, n::Int, settings::ModelSettings)
    d = length(scope)
    k = first(keys(x.data))
    l == 1 && return _productmodel(x, n, settings)
    d == 1 && return ProductNode(_reflectinmodel(x.data[k], settings), k) # best to get rid of this in future (31 in productnode.jl)
    r = ceil(Int, d / 2)
    c = map(1:n) do _
        scope_l, scope_r = scope[1:r], scope[r+1:end]
        comps_l = _productmodel(x[scope_l], scope_l, l-1, n, settings)
        comps_r = _productmodel(x[scope_r], scope_r, l-1, n, settings)
        ProductNode((comps_l, comps_r), [scope_l, scope_r])
    end
    n == 1 ? first(c) : SumNode(c)
end