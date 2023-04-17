
function gmm(n::Int, d::Integer; 
    dtype::Type{<:Real}=Float32, μinit::Symbol=:uniform, Σinit::Symbol=:unit, Σtype::Symbol=:diag, r::Real=0.)

    ps = (; dtype, μinit, Σinit, Σtype, r)
    SumNode([_MvNormal(d; ps...) for _ in 1:n]; dtype=dtype)
end

"""
    spn(d::Int, l::Int, n::Int)
    spn(d::Int, n::Vector{Int})

Implement SumProduct Network for vector data with binary splitting ProductNodes.

# Arguments
- `d::Int`: the dimensionality of feature vector.
- `l::Int`: the number of SumNode layers in the network. 
    E.g., l= 2 means no have SumNode -> ProductNode -> SumNode -> leaf.
- `n::Int`: the number of SumNode components (same for each layer).
- `n::Vector{Int}`: the number of SumNode components (specified for each layer)
# Examples
```julia-repl
julia> spn(10, 2, 2)
SumNode
  ├── ProductNode
  │     ├── SumNode
  │     │     ├── _MvNormal
  │     │     ╰── _MvNormal
  │     ╰── SumNode
  │           ├── _MvNormal
  │           ╰── _MvNormal
  ╰── ProductNode
        ├── SumNode
        │     ├── _MvNormal
        │     ╰── _MvNormal
        ╰── SumNode
              ├── _MvNormal
              ╰── _MvNormal
```

"""
function spn(d::Int, l::Int, n::Int)
    n == 1 && l == 1 && return _MvNormal(d)
    l == 1 && return SumNode(map(_->_MvNormal(d), 1:n)...)

    n == 1 && d == 1 && return _MvNormal(d)
    d == 1 && return SumNode(map(_->_MvNormal(d), 1:n)...)
    
    r = ceil(Int, d / 2)
    comp_sum = map(1:n) do _
        comp_prod = [spn(r, l-1, n), spn(d-r, l-1, n)]
        ProductNode(comp_prod...)
    end
    length(comp_sum) == 1 ? first(comp_sum) : SumNode(comp_sum...)
end

function spn(d::Int, n::Vector{Int})
    length(n) == 1 && n[1] == 1 && return _MvNormal(d)
    length(n) == 1 && return SumNode(map(_->_MvNormal(d), 1:n[1])...)

    d == 1 && n[1] == 1 && return _MvNormal(d)
    d == 1 && return SumNode(map(_->_MvNormal(d), 1:n[1])...)

    comp_sum = map(1:n[1]) do _
        r = ceil(Int, d / 2)
        comp_prod = [spn(r, n[2:end]), spn(d-r, n[2:end])]
        length(comp_prod) == 1 ? first(comp_prod) : ProductNode(comp_prod...)
    end
    length(comp_sum) == 1 ? first(comp_sum) : SumNode(comp_sum...)
end

function setmixture(nb::Int, ni::Int, d::Int; cdist::Function=()->_Poisson(),
    dtype::Type{<:Real}=Float64, μinit::Symbol=:uniform, Σinit::Symbol=:unit, Σtype::Symbol=:full, r::Real=0.)

    ps = (; dtype, μinit, Σinit, Σtype, r)
    fdist() = ni > 1 ? gmm(ni, d; ps...) : _MvNormal(d; ps...)
    
    components = map(1:nb) do _
        pc = cdist()
        pf = fdist()
        SetNode(pf, pc)
    end
    SumNode(components; dtype=dtype)
end

function sharedsetmixture(nb::Int, nis::Int, nin::Int, d::Int; cdist::Function=()->_Poisson(), 
    dtype::Type{<:Real}=Float64, μinit::Symbol=:uniform, Σinit::Symbol=:unit, Σtype::Symbol=:full, r::Real=0.)
    ps = (; dtype, μinit, Σinit, Σtype, r)

    sharedcomps = [_MvNormal(d; ps...) for _ in 1:nis]
    bagcomps = map(1:nb) do _
        pc = cdist()
        nonsharedcomps = [_MvNormal(d; ps...) for _ in 1:nin]
        pf = SumNode([nonsharedcomps; sharedcomps]; dtype)
        SetNode(pf, pc)
    end
    SumNode(bagcomps; dtype)
end
