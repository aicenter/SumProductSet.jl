using SumProductSet, Test, Distributions, Flux
using PoissonRandom

import Mill

@testset "SetNode --- forward" begin
	ndims = 2
    ninst = 10
    nbags = 10
    bagsizes = [pois_rand(3) for _ in 1:nbags]
    bagids = [rand(1:ninst, bagsizes[i]) for i in 1:nbags]

	x = randn(Float32, ndims, ninst)
    BN = Mill.BagNode(x, bagids)

    m = SetNode(SumProductSet.MvNormal(ndims), SumProductSet.Poisson())

	@test !isnothing(SumProductSet.logpdf(m, BN))
    @test length(SumProductSet.logpdf(m, BN)) == nbags 
end


@testset "SetNode --- rand sampling" begin
    ndims = 2
    nobs = 100
    m = SetNode(SumProductSet.MvNormal(ndims), SumProductSet.Poisson())

    @test typeof(rand(m, nobs)) <: Mill.BagNode
    @test Mill.numobs(rand(m, nobs)) == nobs
end


@testset "SetNode --- integration with Flux" begin
    ndims = 2
    nobs = 100
	m = SetNode(SumProductSet.MvNormal(ndims), SumProductSet.Poisson())
    ps = Flux.params(m)

    @test !isempty(ps)
    x = rand(m, nobs)
    @test !isnothing(gradient(() -> sum(SumProductSet.logpdf(m, x)), ps))
end