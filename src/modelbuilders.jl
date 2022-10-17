
function gmm(n::Int, d::Int; 
    dtype::Type{<:Real}=Float64, μinit::Symbol=:uniform, Σinit::Symbol=:unit, Σtype::Symbol=:full)

    ps = (; dtype, μinit, Σinit, Σtype)
    SumNode([_MvNormal(d; ps...) for _ in 1:n]; dtype=dtype)
end

function setmixture(nb::Int, ni::Int, d::Int; 
    dtype::Type{<:Real}=Float64, μinit::Symbol=:uniform, Σinit::Symbol=:unit, Σtype::Symbol=:full)

    ps = (; dtype, μinit, Σinit, Σtype)
    fdist() = ni > 1 ? gmm(ni, d; ps...) : _MvNormal(d; ps...)
    
    components = map(1:nb) do _
        pc = _Poisson()
        pf = fdist()
        SetNode(pf, pc)
    end
    SumNode(components; dtype=dtype)
end

function sharedsetmixture(nb::Int, nis::Int, nin::Int, d::Int;
    dtype::Type{<:Real}=Float64, μinit::Symbol=:uniform, Σinit::Symbol=:unit, Σtype::Symbol=:full)
    ps = (; dtype, μinit, Σinit, Σtype)

    sharedcomps = [_MvNormal(d; ps...) for _ in 1:nis]
    bagcomps = map(1:nb) do _
        pc = _Poisson()
        nonsharedcomps = [_MvNormal(d; ps...) for _ in 1:nin]
        pf = SumNode([nonsharedcomps; sharedcomps]; dtype)
        SetNode(pf, pc)
    end
    SumNode(bagcomps; dtype)
end
