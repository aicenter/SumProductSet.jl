

# shallow mixture
function setmixture(n::Int)
    components = [SetNode(_Poisson(1.))]
    SumNode(components)
end