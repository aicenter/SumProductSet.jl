
function gmm(n::Int, d::Int; 
    dtype::Type{<:Real}=Float64, μinit::Symbol=:uniform, Σinit::Symbol=:unit, Σtype::Symbol=:full, r::Real=0.)

    ps = (; dtype, μinit, Σinit, Σtype, r)
    SumNode([_MvNormal(d; ps...) for _ in 1:n]; dtype=dtype)
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
