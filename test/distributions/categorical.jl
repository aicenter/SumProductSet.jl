using SumProductSet, Test, Distributions, Flux

@testset "_Categorical --- logpdf forward" begin
	n = 10
    m = _Categorical(n)
    x1 = rand(1:n, 100)
    x2 = 6

    @test size(logpdf(m, x1)) == size(x1)
    @test size(logpdf(m, x2)) == size(x2)
end

@testset "_Categorical --- rand sampling" begin
    n = 5
    m = _Categorical(n)
    @test length(rand(m)) == 1
    nobs = 20
    @test length(rand(m, nobs)) == nobs
end

@testset "_Categorical --- integration with Flux" begin
    n = 10
	m = _Categorical(n)
	ps = Flux.params(m)

    @test !isempty(ps)
    x = rand(1:n, 20)
    @test !isnothing(gradient(() -> sum(logpdf(m, x)), ps))

end