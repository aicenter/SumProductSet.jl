

function setmixture(nb::Int, ni::Int, d::Int)

    components = map(1:nb) do _
        pc = _Poisson(log(3.))
        pf = SumNode([_MvNormal(d) for _ in 1:ni])
        pset = SetNode(pf, pc)
    end
    SumNode(components)
end