"""
    x - one Mill sample
    n - number of classes/cluster
    fm - leaf distribution for continuous data. 
        Should be function d -> dist(d), whered is dimenion of feature vector in leaves.
"""
function reflectinmodel(x, n::Int, fm=d->gmm(2, d;), cdist=()->_Poisson(); dtype::Type{<:Real}=Float64)
    SumNode([_reflectinmodel(x, fm, cdist; dtype) for _ in 1:n]; dtype=dtype)
end

function _reflectinmodel(x::Mill.BagNode, fm, cdist; dtype)
    fdist = _reflectinmodel(x.data, fm, cdist; dtype)
    SetNode(fdist, cdist())
end

function _reflectinmodel(x::Mill.ProductNode, fm, cdist; dtype)
    ks = keys(x.data)
    m = tuple([_reflectinmodel(x.data[k], fm, cdist; dtype) for (i, k) in enumerate(ks)]...)
    ProductNode(m)
end

_reflectinmodel(x::Mill.ArrayNode, fm, cdist; dtype) = _reflectinmodel(x.data, fm, cdist; dtype)
_reflectinmodel(x::Flux.OneHotArray, fm, cdist; dtype) = _Categorical(size(x, 1))
_reflectinmodel(x::AbstractArray, fm, cdist; dtype) = fm(size(x, 1))



