using SumProductSet, Test, Distributions, Flux
using PoissonRandom
using StatsBase: nobs

import Mill

@testset "SetNode --- forward" begin
	d = 2
    ninst = 10
    nbags = 10
    bagsizes = [pois_rand(3) for _ in 1:nbags]
    bagids = [rand(1:ninst, bagsizes[i]) for i in 1:nbags]

	x = randn(d, ninst)
    AN = Mill.ArrayNode(x)
    BN = Mill.BagNode(AN, bagids)

    m = SetNode(_MvNormal(d), _Poisson(1.))

	@test logpdf(m, BN) != nothing
    @test length(logpdf(m, BN)) == nbags 
	@test length(m) == 2
end


@testset "SetNode --- rand sampling" begin
    d = 2
	m = SetNode(_MvNormal(d), _Poisson(1.))

    n = 10
    @test typeof(rand(m, n)) <: Mill.BagNode
    @test nobs(rand(m, n)) == n

end


@testset "SetNode --- integration with Flux" begin
    d = 2
	m = SetNode(_MvNormal(d), _Poisson(1.))
    ps = Flux.params(m)

    @test !isempty(ps)
    x = rand(m, 10)
    @test gradient(() -> sum(logpdf(m, x)), ps) != nothing
end