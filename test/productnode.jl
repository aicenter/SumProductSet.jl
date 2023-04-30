using SumProductSet, Test, Distributions, Flux
import Mill

@testset "ProductNode --- forward" begin
    d1 = 9
    d2 = 11
    n = 15
    x = randn(d1 + d2, n)
    m = ProductNode((SumProductSet.MvNormal(d1), SumProductSet.MvNormal(d2)))

    @test !isnothing(SumProductSet.logpdf(m, x))
    @test SumProductSet.logpdf(m, x) ≈ SumProductSet.logpdf(m[1], x[1:d1, :]) + SumProductSet.logpdf(m[2], x[d1+1:d1+d2, :])
end


# @testset "ProductNode --- rand sampling" begin
# end


@testset "ProductNode --- integration with Flux" begin
    d1 = 9
    d2 = 11
    n = 15
    x = randn(d1 + d2, n)

    m = ProductNode((SumProductSet.MvNormal(d1), SumProductSet.MvNormal(d2)))
    ps = Flux.params(m)

    @test !isempty(ps)
    @test !isnothing(gradient(() -> sum(SumProductSet.logpdf(m, x)), ps))
end

@testset "ProductNode --- integration with Mill" begin
    # Integration with Mill.ArrayNode
    d1 = 9
    d2 = 11
    n = 15
    x = Mill.ArrayNode(randn(d1 + d2, n))
    m = ProductNode((SumProductSet.MvNormal(d1), SumProductSet.MvNormal(d2)))

    @test !isnothing(SumProductSet.logpdf(m, x))
    @test SumProductSet.logpdf(m, x) ≈ SumProductSet.logpdf(m[1], x.data[1:d1, :]) + SumProductSet.logpdf(m[2], x.data[d1+1:d1+d2, :])

    # Integration with Mill.ProductNode
    d1 = 9
    d2 = 11
    n = 15
    x = Mill.ProductNode((randn(d1, n), randn(d2, n)))
    m = ProductNode((SumProductSet.MvNormal(d1), SumProductSet.MvNormal(d2)))
    @test !isnothing(SumProductSet.logpdf(m, x))
    @test SumProductSet.logpdf(m, x) ≈ SumProductSet.logpdf(m[1], x.data[1]) + SumProductSet.logpdf(m[2], x.data[2])

end