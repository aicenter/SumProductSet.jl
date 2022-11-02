
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