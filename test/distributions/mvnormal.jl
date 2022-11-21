
const grid = Iterators.product(
    [1, 2, 11, 50],
    [Float32, Float64],
    [:uniform, :randn],
    [:unit, :randlow, :randhigh],
    [:full, :diag],
    [0., 1.])

@testset "_MvNormal --- constructors" begin
    
    for (d, dtype, μinit, Σinit, Σtype, r) in grid
        ps = (; dtype, μinit, Σinit, Σtype, r)
        # @show d, ps
        m = _MvNormal(d; ps...)
        @test !isnothing(m)
        @test length(m) == d
    end
end

@testset "_MvNormal --- initialization and logpdf forward" begin
    
    n = 20
    for (d, dtype, μinit, Σinit, Σtype, r) in grid
        ps = (; dtype, μinit, Σinit, Σtype, r)
        m = _MvNormal(d; ps...)

        x1 = randn(dtype, d, n)
        x2 = randn(dtype, d)

        @test length(SumProductSet.logpdf(m, x1)) == size(x1, 2)
        @test length(SumProductSet.logpdf(m, x2)) == size(x2, 2)
        @test typeof(sum(SumProductSet.logpdf(m, x1))) == dtype
    end
end

@testset "_MvNormal --- rand sampling" begin

    n = 20
    for (d, dtype, μinit, Σinit, Σtype, r) in grid
        ps = (; dtype, μinit, Σinit, Σtype, r)
        m = _MvNormal(d; ps...)

        @test size(rand(m)) == (d,)
        @test size(rand(m, n)) == (d, n)
        @test eltype(rand(m, n)) == dtype
    end
end

@testset "_MvNormal correctness" begin
    μ = [-3., 11]
    Σ = [5. 3; 3 7]
    n = 10000
    m1 = MvNormal(μ, Σ)
    m2 = _MvNormalParams(μ, Σ)
    x1 = rand(m1, n) 
    x2 = rand(m2, n)

    @test Distributions.logpdf(m1, x1) ≈ SumProductSet.logpdf(m2, x1)
    @test Distributions.logpdf(m1, x2) ≈ SumProductSet.logpdf(m2, x2)
end

@testset "_MvNormal --- integration with Flux" begin

    for (d, dtype, μinit, Σinit, Σtype, r) in grid
        ps = (; dtype, μinit, Σinit, Σtype, r)
        m = _MvNormal(d; ps...)

        ps = Flux.params(m)
        @test !isempty(ps)
        x = rand(m, 10)
        @test !isnothing( gradient(() -> sum(SumProductSet.logpdf(m, x)), ps) )
    end
end
