using SumProductSet, Test, Distributions, StatsBase
import Mill

const sm_grid = Iterators.product(
    [2, 10],
    [1, 3],
    [1, 2, 11, 50],
    [Float32, Float64],
    [:uniform, :randn],
    [:unit, :randlow, :randhigh],
    [:full, :diag])

@testset "setmixture --- constructor" begin
    for (nb, ni, d, dtype, μinit, Σinit, Σtype) in sm_grid
        ps = (; dtype, μinit, Σinit, Σtype)
        m = setmixture(nb, ni, d; ps...)
        @test !isnothing(m)
        @test length(m) == d
    end
end

@testset "setmixture --- rand sampling" begin
    npts = 20 
    for (nb, ni, d, dtype, μinit, Σinit, Σtype) in sm_grid
        ps = (; dtype, μinit, Σinit, Σtype)
        m = setmixture(nb, ni, d; ps...)
        @test typeof(rand(m)) <: Mill.BagNode
        @test typeof(rand(m, npts)) <: Mill.BagNode  

        @test nobs(rand(m)) == 1
        @test nobs(rand(m, npts)) == npts

        @test eltype(rand(m).data.data) <: Union{dtype, Missing}
        @test eltype(rand(m, npts).data.data) <: Union{dtype, Missing}
    end
end

@testset "setmixture --- logpdf forward" begin
    npts = 20 
    for (nb, ni, d, dtype, μinit, Σinit, Σtype) in sm_grid
        ps = (; dtype, μinit, Σinit, Σtype)
        m = setmixture(nb, ni, d; ps...)
        x1 = rand(m)
        x2 = rand(m, npts)

        @test !isnothing(SumProductSet.logpdf(m, x1))
        @test !isnothing(SumProductSet.logpdf(m, x2))
    end
end

@testset "hierarchical model -- logpdf forward" begin
    d1 = 9
    d2 = 11
    pdist = ()->(SumProductSet.MvNormal(d1), SumProductSet.MvNormal(d2))
    cdist = ()-> SumProductSet.Poisson()

    prodmodel = ()->ProductNode(pdist())
    setmodel  = ()->SetNode(prodmodel(), cdist())

    m = SumNode([setmodel() for _ in 1:3])

    n = 30
    pn = Mill.ProductNode((randn(d1, n), randn(d2, n)))
    bn = Mill.BagNode(pn, [1:5, 6:15, 16:16, 17:30])

    @test !isnothing(SumProductSet.logpdf(m, bn))
end
