using SumProductSet, Test, Distributions, Flux
import Mill

@testset "ProductNode --- forward" begin

    d1 = 9
    d2 = 11
    n = 15
    x = randn(d1 + d2, n)
    m = ProductNode((_MvNormal(d1), _MvNormal(d2)))

    @test !isnothing(logpdf(m, x))
    @test logpdf(m, x) ≈ logpdf(m[1], x[1:d1, :]) + logpdf(m[2], x[d1+1:d1+d2, :])
end


# @testset "ProductNode --- rand sampling" begin
# end


@testset "ProductNode --- integration with Flux" begin
    d1 = 9
    d2 = 11
    n = 15
    x = randn(d1 + d2, n)

    m = ProductNode((_MvNormal(d1), _MvNormal(d2)))
    ps = Flux.params(m)

    @test !isempty(ps)
    @test !isnothing(gradient(() -> sum(logpdf(m, x)), ps))
end

@testset "ProductNode --- integration with Mill" begin

    # Integration with Mill.ArrayNode
    d1 = 9
    d2 = 11
    n = 15
    x = Mill.ArrayNode(randn(d1 + d2, n))
    m = ProductNode((_MvNormal(d1), _MvNormal(d2)))
    
    @test !isnothing(logpdf(m, x))
    @test logpdf(m, x) ≈ logpdf(m[1], x.data[1:d1, :]) + logpdf(m[2], x.data[d1+1:d1+d2, :])

    # Integration with Mill.ProductNode
    d1 = 9
    d2 = 11
    n = 15
    x = Mill.ProductNode((randn(d1, n), randn(d2, n)))
    m = ProductNode((_MvNormal(d1), _MvNormal(d2)))
    @test !isnothing(logpdf(m, x))
    @test logpdf(m, x) ≈ logpdf(m[1], x.data[1]) + logpdf(m[2], x.data[2])

end