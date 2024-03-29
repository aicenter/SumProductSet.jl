
const grid = Iterators.product(
    [1, 2, 11, 50],
    [Float32, Float64],
    [:uniform, :randn],
    [:unit, :randlow, :randhigh],
    [:full, :diag],
    [0., 1.])

@testset "MvNormal --- constructors" begin
    for (d, dtype, minit, sinit, stype, r) in grid
        ps = (; dtype, minit, sinit, stype, r)
        # @show d, ps
        m = SumProductSet.MvNormal(d; ps...)
        @test !isnothing(m)
        @test length(m) == d
    end
end

@testset "MvNormal --- initialization and logpdf forward" begin
    n = 20
    for (d, dtype, minit, sinit, stype, r) in grid
        ps = (; dtype, minit, sinit, stype, r)
        m = SumProductSet.MvNormal(d; ps...)

        x1 = randn(dtype, d, n)
        x2 = randn(dtype, d)

        @test length(SumProductSet.logpdf(m, x1)) == size(x1, 2)
        @test length(SumProductSet.logpdf(m, x2)) == size(x2, 2)
        @test typeof(sum(SumProductSet.logpdf(m, x1))) == dtype
    end
end

@testset "MvNormal --- rand sampling" begin
    n = 20
    for (d, dtype, minit, sinit, stype, r) in grid
        ps = (; dtype, minit, sinit, stype, r)
        m = SumProductSet.MvNormal(d; ps...)

        @test size(rand(m)) == (d, 1)
        @test size(rand(m, n)) == (d, n)
        @test typeof(rand(m, n)) <: Mill.ArrayNode
        @test eltype(rand(m, n).data) == dtype
    end
end

@testset "MvNormal --- correctness" begin
    μ = [-3., 11]
    Σ = [5. 3; 3 7]
    n = 10000
    m1 = Distributions.MvNormal(μ, Σ)
    m2 = SumProductSet.MvNormalParams(μ, Σ)
    x1 = rand(m1, n) 
    x2 = rand(m2, n)

    @test Distributions.logpdf(m1, x1)[:] ≈ SumProductSet.logpdf(m2, x1)[:]
    @test Distributions.logpdf(m1, x2.data)[:] ≈ SumProductSet.logpdf(m2, x2)[:]
end

@testset "MvNormal --- integration with Flux" begin
    for (d, dtype, minit, sinit, stype, r) in grid
        ps = (; dtype, minit, sinit, stype, r)
        m = SumProductSet.MvNormal(d; ps...)

        ps = Flux.params(m)
        @test !isempty(ps)
        x = rand(m, 10)
        @test !isnothing(gradient(() -> sum(SumProductSet.logpdf(m, x)), ps))
    end
end
