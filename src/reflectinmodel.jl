"""
    Build mixture model of HMILL data.
    x - one HMLL sample
    n - number of mixture components in root node 
    f_{type} - leaf distribution for {type} variable
        Should be function d -> dist(d), where d is leaf data dimension
"""

function reflectinmodel(x, n::Int; f_cont=d->gmm(2, d), f_cat=d->_Categorical(d), f_card=()->_Poisson(), dtype::Type{<:Real}=Float64)
    SumNode([_reflectinmodel(x; f_cont=f_cont, f_cat=f_cat, f_card=f_card, dtype=dtype) for _ in 1:n]; dtype=dtype)
end

function _reflectinmodel(x::Mill.BagNode; f_cont, f_cat, f_card, dtype)
    fdist = _reflectinmodel(x.data; f_cont=f_cont, f_cat=f_cat, f_card=f_card, dtype=dtype)
    SetNode(fdist, f_card())
end

function _reflectinmodel(x::Mill.ProductNode{<:NamedTuple}; f_cont, f_cat, f_card, dtype)
    ks = keys(x.data)
    ms = NamedTuple(k => _reflectinmodel(x.data[k]; f_cont=f_cont, f_cat=f_cat, f_card=f_card, dtype=dtype) for k in ks)
    ProductNode(ms)
end

_reflectinmodel(x::Mill.ArrayNode; f_cont, f_cat, f_card, dtype) = _reflectinmodel(x.data; f_cont=f_cont, f_cat=f_cat, f_card=f_card, dtype=dtype)
_reflectinmodel(x::Flux.OneHotArray; f_cont, f_cat, f_card, dtype) = f_cat(size(x, 1))
_reflectinmodel(x::AbstractArray; f_cont, f_cat, f_card, dtype) = f_cont(size(x, 1))
