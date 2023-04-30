
@testset "Categorical --- logpdf forward" begin
	n = 10
    m = SumProductSet.Categorical(n)
    x1 = rand(1:n, 100)
    x2 = 6

    @test size(SumProductSet.logpdf(m, x1)) == size(x1)
    @test size(SumProductSet.logpdf(m, x2)) == size(x2)
end

@testset "Categorical --- rand sampling" begin
    n = 5
    m = SumProductSet.Categorical(n)
    @test length(rand(m)) == 1
    nobs = 20
    @test length(rand(m, nobs)) == nobs
end

@testset "Categorical --- integration with Flux" begin
    n = 10
	m = SumProductSet.Categorical(n)
	ps = Flux.params(m)

    @test !isempty(ps)
    x = rand(1:n, 20)
    @test !isnothing(gradient(() -> sum(SumProductSet.logpdf(m, x)), ps))
end

@testset "Categorical --- correctness" begin
    p = [0.1, 0.3, 0.15, 0.45]
    n = 100
    m1 = Distributions.Categorical(p)
    m2 = SumProductSet.Categorical(log.(p))
    x1 = rand(m1, n) 
    x2 = rand(m2, n)

    @test Distributions.logpdf.(m1, x1) ≈ SumProductSet.logpdf(m2, x1)
    @test Distributions.logpdf.(m1, x2) ≈ SumProductSet.logpdf(m2, x2)
end

@testset "Categorical --- integration with OneHotArrays" begin
    n = 20
    c = 10
	m = SumProductSet.Categorical(c)
    ps = Flux.params(m)

    x = rand(m)
    x_oh = Flux.onehot(x, 1:c)
    @test size(SumProductSet.logpdf(m, x)) == ()
    @test !isnothing(gradient(() -> sum(SumProductSet.logpdf(m, x)), ps))

    x = rand(m, 20)
    x_oh = Flux.onehotbatch(x, 1:c)
    @test size(SumProductSet.logpdf(m, x)) == (n,)
    @test !isnothing(gradient(() -> sum(SumProductSet.logpdf(m, x)), ps))
end
