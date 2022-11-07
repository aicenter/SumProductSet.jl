
@testset "_Poisson --- logpdf forward" begin
	m = _Poisson(log(5))
    xs = [rand(1:20, 10), 2, 100., 0]

    for x in xs
        @test size(SumProductSet.logpdf(m, x)) == size(x)
    end

    # test for MvPoisson (independent), but MvPoisson construction is disabled for now
    # @test SumProductSet.logpdf(_Poisson(log.([10, 5])), xs[1]) ≈ SumProductSet.logpdf(_Poisson(log(10)), xs[1]) + SumProductSet.logpdf(_Poisson(log(5)), xs[1])

end

@testset "_Poisson --- rand sampling" begin
    m = _Poisson(log(6))
    @test length(rand(m)) == length(m.logλ)
end

@testset "_Poisson --- integration with Flux" begin

	m = _Poisson(log(5))
    truegrad(logλ, x) = -exp.(logλ) .+ x  # d(SumProductSet.logpdf(Poiss(x ; logλ))) / d(logλ)
	ps = Flux.params(m)

    @test !isempty(ps)
    x = rand(1:20, 10)
    @test gradient(() -> sum(SumProductSet.logpdf(m, x)), ps) != nothing
    gs = gradient(() -> sum(SumProductSet.logpdf(m, x)), ps)
    for p in ps
        @test gs[p] ≈ mapreduce(xi->truegrad(m.logλ, xi), +, x)
    end
end

@testset "_Poisson --- correctness" begin
    λ = 5
    n = 100
    m1 = Poisson(λ)
    m2 = _Poisson(log(λ))
    x1 = rand(m1, n) 
    x2 = rand(m2, n)

    @test Distributions.logpdf.(m1, x1) ≈ SumProductSet.logpdf(m2, x1)
    @test Distributions.logpdf.(m1, x2) ≈ SumProductSet.logpdf(m2, x2)
end