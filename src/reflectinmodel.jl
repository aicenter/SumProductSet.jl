"""
    reflectinmodel(x, n; kwargs...)
Build mixture model of HMILL data.
x - one HMLL sample
n - number of mixture components in root node 
f_{type} - leaf distribution for {type} variable
    Should be function d -> distribution(d), where d is dimension of data leaf

# Examples

```julia-repl
z = Mill.ArrayNode([0. 1; 2 3]);
x = Mill.ProductNode(a=z, b=z, c=z, d=z);
reflectinmodel(x, 2; depth_prod=2, n_prod_mix=2)
```
"""
reflectinmodel(x, n::Integer; n_set_mix::Int=1, depth_prod::Int=1, n_prod_mix::Int=1, f_cont=d->gmm(2, d), f_cat=d->Categorical(d), f_disc=d->Geometric(d), f_card=()->Poisson(), dtype::Type{<:Real}=Float32) =
    _reflectinmodel(x; (; n_set_mix, depth_prod, n_prod_mix=vcat(n, repeat([n_prod_mix], 99)), f_cont, f_cat, f_disc, f_card, dtype)...)

function _reflectinmodel(x::Mill.ProductNode; kwargs...)
    n_prod_mix = kwargs[:n_prod_mix]
    depth_prod = kwargs[:depth_prod]

    if depth_prod == 1
        _productmodel(x,                      n_prod_mix[1]; filter(p->p[1]!="n_prod_mix", kwargs)..., n_prod_mix=n_prod_mix[2:end])
    else 
        _productmodel(x, keys(x), depth_prod, n_prod_mix[1]; filter(p->p[1]!="n_prod_mix", kwargs)..., n_prod_mix=n_prod_mix[2:end])   
    end
end

function _reflectinmodel(x::Mill.BagNode; kwargs...)
    n_set_mix = kwargs[:n_set_mix]
    f_card = kwargs[:f_card]
    f_inst = ()->_reflectinmodel(x.data; kwargs...)
    n_set_mix == 1 ? SetNode(f_inst(), f_card()) : SumNode(map(_->SetNode(f_inst(), f_card()), 1:n_set_mix))
end

_reflectinmodel(x::Mill.ArrayNode; kwargs...) = _reflectinmodel(x.data; kwargs...)

_reflectinmodel(x::OneHotArray;           kwargs...)                     = kwargs[:f_cat ](size(x, 1))
_reflectinmodel(x::MaybeHotArray;         kwargs...)                     = kwargs[:f_cat ](size(x, 1))
_reflectinmodel(x::Array{T};              kwargs...) where T <: Real     = kwargs[:f_cont](size(x, 1))
_reflectinmodel(x::Array{Maybe{T}};       kwargs...) where T <: Real     = kwargs[:f_cont](size(x, 1))
_reflectinmodel(x::NGramMatrix{T};        kwargs...) where T <: Sequence = kwargs[:f_disc](size(x, 1))
_reflectinmodel(x::NGramMatrix{Maybe{T}}; kwargs...) where T <: Sequence = kwargs[:f_disc](size(x, 1))


function _productmodel(x, n::Int; kwargs...)
    k = keys(x.data)
    c = map(_->ProductNode(mapreduce(k->_reflectinmodel(x.data[k]; kwargs...), vcat, k), reduce(vcat, k)), 1:n)
    n == 1 ? first(c) : SumNode(c)
end
function _productmodel(x, scope::NTuple{N, Symbol}, l::Int, n::Int; kwargs...) where {N}
    d = length(scope)
    k = first(keys(x.data))
    l == 1 && return _productmodel(x, n; kwargs...)
    d == 1 && return ProductNode(_reflectinmodel(x.data[k]; kwargs...), k) # best to get rid of this in future (31 in productnode.jl)
    r = ceil(Int, d / 2)
    c = map(1:n) do _
        scope_l, scope_r = scope[1:r], scope[r+1:end]
        comps_l = _productmodel(x[scope_l], scope_l, l-1, n; kwargs...)
        comps_r = _productmodel(x[scope_r], scope_r, l-1, n; kwargs...)
        ProductNode([comps_l, comps_r], [scope_l, scope_r])
    end
    SumNode(c)
end
