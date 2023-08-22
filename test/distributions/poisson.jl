
@testset "Poisson --- logpdf forward" begin
    ndim = 1
	m = SumProductSet.Poisson(ndim)
    xs = [rand(0:20, ndim, 100), 2, 100, 0]

    for x in xs
        @test !isnothing(SumProductSet.logpdf(m, x))
    end

    ndim = 10
    m = SumProductSet.Poisson(ndim)
    xs = rand(0:20, ndim, 100)
    @test !isnothing(SumProductSet.logpdf(m, xs))
end

@testset "Poisson --- rand sampling" begin
    ndim = 1
    m = SumProductSet.Poisson(ndim)
    @test typeof(rand(m)) <: Mill.ArrayNode
    @test length(rand(m).data) == length(m.lograte)
end

@testset "Poisson --- rrule test" begin
    ndims = 10
    nobs = 100
	m = SumProductSet.Poisson(ndims)
    x = rand(m, nobs)

    test_rrule(SumProductSet._logpdf_poisson, m.lograte, x.data ⊢ NoTangent();
                check_inferred=true, rtol = 1.0e-9, atol = 1.0e-9)
end

@testset "Poisson --- integration with Flux" begin
    ndim = 10
	m = SumProductSet.Poisson(ndim)
    ps = Flux.params(m)

    @test !isempty(ps)
    x = rand(0:20, ndim, 100)
    @test !isnothing(gradient(() -> sum(SumProductSet.logpdf(m, x)), ps))
end

@testset "Poisson --- correctness" begin
    λ = 5
    n = 100
    m1 = Distributions.Poisson(λ)
    m2 = SumProductSet.Poisson(log(λ))
    x1 = rand(m1, n) 
    x2 = rand(m2, n)

    @test Distributions.logpdf.(m1, x1)[:] ≈ SumProductSet.logpdf(m2, x1)[:]
    @test Distributions.logpdf.(m1, x2.data)[:] ≈ SumProductSet.logpdf(m2, x2)[:]
end