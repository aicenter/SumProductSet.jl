
function gmm(n::Int, d::Integer; 
    dtype::Type{<:Real}=Float32, minit::Symbol=:uniform, sinit::Symbol=:unit, stype::Symbol=:diag, r::Real=0.)

    ps = (; dtype, minit, sinit, stype, r)
    SumNode([MvNormal(d; ps...) for _ in 1:n]; dtype=dtype)
end

"""
    spn(d::Int, l::Int, n::Int)
    spn(d::Int, n::Vector{Int})

Implement SumProduct Network for vector data with binary splits at ProductNodes.

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
  │     │     ├── MvNormal
  │     │     ╰── MvNormal
  │     ╰── SumNode
  │           ├── MvNormal
  │           ╰── MvNormal
  ╰── ProductNode
        ├── SumNode
        │     ├── MvNormal
        │     ╰── MvNormal
        ╰── SumNode
              ├── MvNormal
              ╰── MvNormal
```

"""
function spn(d::Int, l::Int, n::Int; leaf::Function=d->MvNormal(d))
    (l == 1 || d == 1) && n == 1 && return leaf(d)
    (l == 1 || d == 1) && return SumNode(map(_->leaf(d), 1:n))

    r = ceil(Int, d / 2)
    comp_sum = map(1:n) do _
        prod_comp = (spn(r, l-1, n; leaf=leaf), spn(d-r, l-1, n; leaf=leaf))
        prod_dims = [1:r, r+1:d]
        ProductNode(prod_comp, prod_dims)
    end
    length(comp_sum) == 1 ? first(comp_sum) : SumNode(comp_sum)
end

function spn(d::Int, n::Vector{Int}; leaf::Function=d->MvNormal(d))
    (length(n) == 1 || d == 1) && first(n) == 1 && return leaf(d)
    (length(n) == 1 || d == 1) && return SumNode(map(_->leaf(d), 1:first(n)))

    r = ceil(Int, d / 2)
    sum_comp = map(1:first(n)) do _
        prod_comp = (spn(r, n[2:end]; leaf=leaf), spn(d-r, n[2:end]; leaf=leaf))
        prod_dims = [1:r, r+1:d]

        length(prod_comp) == 1 ? first(prod_comp) : ProductNode(prod_comp, prod_dims)
    end
    length(sum_comp) == 1 ? first(sum_comp) : SumNode(sum_comp)
end

"""
    setmixture(nb::Int, ni::Int, d::Int; kwargs...)

Implement a constructor for mixture of `SetNode`s with specifed feature and cardinality distributions.

# Arguments
- `nb::Int`: the number of `SetNode` components in the root mixture.
- `ni::Int`: the number of `Leaf` components in the `SetNodes`'s feature mixture. 
- `d::Int`: the dimensionality of the `Leaf` distribution.
# Examples
```julia-repl
julia> setmixture(2, 3, 4)
SumNode
  ├── SetNode
  │     ├── c: Poisson
  │     ╰── f: SumNode
  │              ├── MvNormal
  │              ├── MvNormal
  │              ╰── MvNormal
  ╰── SetNode
        ├── c: Poisson
        ╰── f: SumNode
                 ├── MvNormal
                 ├── MvNormal
                 ╰── MvNormal
```

"""
function setmixture(nb::Int, ni::Int, d::Int; 
    leaf::Function=d->MvNormal(d), card_dist::Function=()->Poisson(), dtype::Type{<:Real}=Float32)

    feat_dist() = ni > 1 ? SumNode([leaf(d) for _ in 1:ni]; dtype=dtype) : leaf(d)
    
    set_dists = map(1:nb) do _
        pc = card_dist()
        pf = feat_dist()
        SetNode(pf, pc)
    end
    SumNode(set_dists; dtype=dtype)
end

"""
    sharedsetmixture(nb::Int, nis::Int, nin::Int, d::Int; kwargs...)

Implement a constructor for mixture of `SetNode`s with specifed feature and cardinality distributions.
Some of feature components are shared according to chosen constructor parameters.

# Arguments
- `nb::Int`: the number of `SetNode` components in the root mixture.
- `nis::Int`: the number of `Leaf` components in the `SetNodes`'s feature mixture that are
shared with `Leaf` components among all other `SetNode`s on the same hierarchy level. 
- `nis::Int`: the number of `Leaf` components in the `SetNodes`'s feature mixture that are
`not` shared with `Leaf` components among all other `SetNode`s on the same hierarchy level. 
- `d::Int`: the dimensionality of the `Leaf` distribution.
# Examples
```julia-repl
julia> sharedsetmixture(2, 1, 1, 3)
SumNode
  ├── SetNode
  │     ├── c: Poisson
  │     ╰── f: SumNode
  │              ├── MvNormal
  │              ╰── MvNormal
  ╰── SetNode
        ├── c: Poisson
        ╰── f: SumNode
                 ├── MvNormal
                 ╰── MvNormal

julia> m.components[2].feature.components[1] == m.components[1].feature.components[1]
false

julia> m.components[2].feature.components[2] == m.components[1].feature.components[2]
true
```

"""
function sharedsetmixture(nb::Int, nis::Int, nin::Int, d::Int; 
    leaf::Function=d->MvNormal(d), card_dist::Function=()->Poisson(), dtype::Type{<:Real}=Float32)

    scomponents = [leaf(d) for _ in 1:nis]
    set_dists = map(1:nb) do _
        pc = card_dist()
        nscomponents = [leaf(d) for _ in 1:nin]
        pf = SumNode([nscomponents; scomponents]; dtype)
        SetNode(pf, pc)
    end
    SumNode(set_dists; dtype)
end
