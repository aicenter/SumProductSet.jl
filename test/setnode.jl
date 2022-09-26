using SumProductSet, Test, Distributions, Flux
using PoissonRandom
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