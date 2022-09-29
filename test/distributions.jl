using SumProductSet, Test, Distributions, Flux

@testset "_Poisson --- logpdf forward" begin
	m = _Poisson(log(5))
    xs = [rand(1:20, 10), 2, 100., []]

    for x in xs
        @test size(logpdf.(m, x)) == size(x)
    end

    @test logpdf.(_Poisson(log.([10, 5])), xs[1]) ≈ logpdf.(_Poisson(log(10)), xs[1]) + logpdf.(_Poisson(log(5)), xs[1])

end

@testset "_Poisson --- rand sampling" begin
    m = _Poisson(log(6))
    @test length(rand(m)) == length(m.logλ)

    # m = _Poisson(log.([2, 7, 10]))
    # @test length(rand(m)) == length(m.logλ)
end

@testset "_Poisson --- integration with Flux" begin

	m = _Poisson(log(5))
    truegrad(logλ, x) = -exp.(logλ) .+ x  # d(logpdf(Poiss(x ; logλ))) / d(logλ)
	ps = Flux.params(m)

    @test !isempty(ps)
    x = rand(1:20, 10)
    @test gradient(() -> sum(logpdf.(m, x)), ps) != nothing
    gs = gradient(() -> sum(logpdf.(m, x)), ps)
    for p in ps
        @test gs[p] ≈ mapreduce(xi->truegrad(m.logλ, xi), +, x)
    end
end

@testset "_MvNormal --- logpdf forward" begin
    d = 2
	m = _MvNormal(d)
    xs = [randn(d, 10), randn(d)]

    for x in xs
        @test length(logpdf(m, x)) == size(x, 2)
    end

    @test length(m) == d
end

@testset "_MvNormal --- rand sampling" begin
    d = 2
    m = _MvNormal(d)

    @test size(rand(m)) == (d,)
    n = 10
    @test size(rand(m, n)) == (d, n)
end

@testset "_MvNormal correctness" begin
    μ = [-3., 11]
    Σ = [5. 3; 3 7]
    n = 10000
    m1 = MvNormal(μ, Σ)
    m2 = _MvNormalParams(μ, Σ)
    x1 = rand(m1, n) 
    x2 = rand(m2, n)

    @test logpdf(m1, x1) ≈ logpdf(m2, x1)
    @test logpdf(m1, x2) ≈ logpdf(m2, x2)

end

@testset "_MvNormal --- integration with Flux" begin
    d = 2
	m = _MvNormal(d)
	ps = Flux.params(m)

    @test !isempty(ps)
    x = rand(d, 10)
    @test gradient(() -> sum(logpdf(m, x)), ps) != nothing
end