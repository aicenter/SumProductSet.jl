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
function reflectinmodel(x, n::Int; n_set_mix::Int=1, depth_prod::Int=2, n_prod_mix::Int=2,
    f_cont=d->gmm(2, d), f_cat=d->_Categorical(d), f_card=()->_Poisson(), dtype::Type{<:Real}=Float64)

    _reflectinmodel(x; (; n_set_mix, depth_prod, n_prod_mix=vcat(n, repeat([n_prod_mix], 99)), f_cont, f_cat, f_card, dtype)...)
end


function _reflectinmodel(x::Mill.BagNode; kwargs...)
    n_set_mix = kwargs[:n_set_mix]
    @assert n_set_mix > 0
    f_card = kwargs[:f_card]
    f_inst = ()->_reflectinmodel(x.data; kwargs...)
    n_set_mix == 1 ? SetNode(f_inst(), f_card()) : SumNode(map(_->SetNode(f_inst(), f_card()), 1:n_set_mix)...)
end

# simple product
function _build_prod(x, n::Int; kwargs...)
    c = map(1:n) do _
        ms = NamedTuple(map(k -> k => _reflectinmodel(x.data[k]; kwargs...), keys(x.data)))
        ProductNode(ms)
    end
    length(c) == 1 ? first(c) : SumNode(c...)
end

# more complex product
function _build_prod(x, scope::Vector{Symbol}, l::Int, n::Int; kwargs...)
    d = length(scope)
    @show scope, d, n
    l == 1 && return _build_prod(x, n; kwargs...)
    d == 1 && return ProductNode(NamedTuple(map(_->k=>_reflectinmodel(x.data[k]; kwargs...), keys(x.data))...))
    r = ceil(Int, d / 2)
    comp_sum = map(1:n) do _
        scope_l, scope_r = scope[1:r], scope[r+1:end]
        x_l, x_r = x[scope_l], x[scope_r]

        comp_prod_l = _build_prod(x_l, scope_l, l-1, n; kwargs...)
        comp_prod_r = _build_prod(x_r, scope_r, l-1, n; kwargs...)
        comp_prod = Dict(Set(scope_l)=> comp_prod_l, Set(scope_r)=> comp_prod_r)

        ProductNode(comp_prod)
    end
    SumNode(comp_sum...)
end

function _reflectinmodel(x::Mill.ProductNode{<:NamedTuple}; kwargs...)
    n_prod_mix = kwargs[:n_prod_mix]
    depth_prod = kwargs[:depth_prod]
    @assert depth_prod > 0

    if depth_prod == 1
        _build_prod(x, n_prod_mix[1]; filter(p->p[1] != "n_prod_mix", kwargs)..., n_prod_mix=n_prod_mix[2:end])
    else 
        _build_prod(x, collect(keys(x)), depth_prod, n_prod_mix[1]; filter(p->p[1] != "n_prod_mix", kwargs)..., n_prod_mix=n_prod_mix[2:end])   
    end
end

_reflectinmodel(x::Mill.ArrayNode; kwargs...) = _reflectinmodel(x.data; kwargs...)
_reflectinmodel(x::Flux.OneHotArray; kwargs...) = kwargs[:f_cat](size(x, 1))
_reflectinmodel(x::AbstractArray; kwargs...) = kwargs[:f_cont](size(x, 1))
