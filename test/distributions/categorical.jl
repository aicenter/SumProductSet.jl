
@testset "Categorical --- logpdf forward" begin
	ndims = 10
    m = SumProductSet.Categorical(ndims)
    x1 = rand(1:ndims, 1, 100)
    x2 = 6

    @test length(SumProductSet.logpdf(m, x1)) == length(x1)
    @test length(SumProductSet.logpdf(m, x2)) == length(x2)
end

@testset "Categorical --- rand sampling" begin
    ncat = 10
    m = SumProductSet.Categorical(ncat)
    @test typeof(rand(m)) <: Mill.ArrayNode
    @test typeof(rand(m).data) <: Flux.OneHotArrays.OneHotMatrix
    @test size(rand(m)) == (ncat, 1)
    nobs = 100
    @test size(rand(m, nobs)) == (ncat, nobs)
end

@testset "Categorical --- integration with Flux" begin
    ndims = 10
	m = SumProductSet.Categorical(ndims)
	ps = Flux.params(m)

    @test !isempty(ps)
    x = rand(1:ndims, 1, 100)
    @test !isnothing(gradient(() -> sum(SumProductSet.logpdf(m, x)), ps))
end

@testset "Categorical --- correctness" begin
    p = [0.1, 0.3, 0.15, 0.45]  # sum(p)=1
    n = 100
    m1 = Distributions.Categorical(p)
    m2 = SumProductSet.Categorical(log.(p))
    x1 = rand(m1, n) 
    x2 = rand(m2, n)

    @test Distributions.logpdf.(m1, x1)[:] ≈ SumProductSet.logpdf(m2, x1)[:]
    @test Distributions.logpdf.(m1, Flux.onecold(x2.data))[:] ≈ SumProductSet.logpdf(m2, x2)[:]
end

@testset "Categorical --- integration with OneHotArrays" begin
    nobs = 20
    ncat = 10
	m = SumProductSet.Categorical(ncat)
    ps = Flux.params(m)

    x_oh = rand(m)
    @test !isnothing(SumProductSet.logpdf(m, x_oh))
    @test !isnothing(gradient(() -> sum(SumProductSet.logpdf(m, x_oh)), ps))

    x_oh = rand(m, nobs)
    @test !isnothing(SumProductSet.logpdf(m, x_oh))
    @test !isnothing(gradient(() -> sum(SumProductSet.logpdf(m, x_oh)), ps))
end
