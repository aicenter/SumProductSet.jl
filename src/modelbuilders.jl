
function gmm(n::Int, d::Int)
    SumNode([_MvNormal(d) for _ in 1:n])
end


function setmixture(nb::Int, ni::Int, d::Int; fdist=:gmm, covtype=:full)
    f() = if fdist == :gmm
        gmm(ni, d) 
    elseif fdist ∈ [:mvnormal, :MvNormal, :normal, :gaussian]
        _MvNormal(d)
    else
        @error "Unknown fdist $(fdist)"
    end
    
    components = map(1:nb) do _
        pc = _Poisson()
        pf = f()
        SetNode(pf, pc)
    end
    SumNode(components)
end
