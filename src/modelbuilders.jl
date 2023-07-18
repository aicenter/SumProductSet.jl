
function gmm(n::Int, d::Integer; 
    dtype::Type{<:Real}=Float32, minit::Symbol=:uniform, sinit::Symbol=:unit, stype::Symbol=:diag, r::Real=0.)

    ps = (; dtype, minit, sinit, stype, r)
    SumNode([MvNormal(d; ps...) for _ in 1:n]; dtype=dtype)
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
        prod_comp = [spn(r, l-1, n), spn(d-r, l-1, n)]
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
        prod_comp = [spn(r, n[2:end]), spn(d-r, n[2:end])]
        prod_dims = [1:r, r+1:d]

        length(prod_comp) == 1 ? first(prod_comp) : ProductNode(prod_comp, prod_dims)
    end
    length(sum_comp) == 1 ? first(sum_comp) : SumNode(sum_comp)
end

function setmixture(nb::Int, ni::Int, d::Int; cdist::Function=()->Poisson(),
    dtype::Type{<:Real}=Float64, minit::Symbol=:uniform, sinit::Symbol=:unit, stype::Symbol=:full, r::Real=0.)

    ps = (; dtype, minit, sinit, stype, r)
    fdist() = ni > 1 ? gmm(ni, d; ps...) : MvNormal(d; ps...)
    
    components = map(1:nb) do _
        pc = cdist()
        pf = fdist()
        SetNode(pf, pc)
    end
    SumNode(components; dtype=dtype)
end

function sharedsetmixture(nb::Int, nis::Int, nin::Int, d::Int; cdist::Function=()->Poisson(), 
    dtype::Type{<:Real}=Float64, minit::Symbol=:uniform, sinit::Symbol=:unit, stype::Symbol=:full, r::Real=0.)
    ps = (; dtype, minit, sinit, stype, r)

    sharedcomps = [MvNormal(d; ps...) for _ in 1:nis]
    bagcomps = map(1:nb) do _
        pc = cdist()
        nonsharedcomps = [MvNormal(d; ps...) for _ in 1:nin]
        pf = SumNode([nonsharedcomps; sharedcomps]; dtype)
        SetNode(pf, pc)
    end
    SumNode(bagcomps; dtype)
end
