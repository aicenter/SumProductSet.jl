using SumProductSet, Test, Distributions, Flux
import Mill

@testset "ProductNode --- forward" begin
    d1 = 9
    d2 = 11
    nobs = 100
    x = randn(Float32, d1 + d2, nobs)
    m = ProductNode([SumProductSet.MvNormal(d1), SumProductSet.MvNormal(d2)], [1:d1, d1+1:d1+d2])

    @test !isnothing(SumProductSet.logpdf(m, x))
    @test length(SumProductSet.logpdf(m, x)) == nobs
    @test SumProductSet.logpdf(m, x) ≈ SumProductSet.logpdf(m.components[1], x[1:d1, :]) + 
                                       SumProductSet.logpdf(m.components[2], x[d1+1:d1+d2, :])
end


# @testset "ProductNode --- rand sampling" begin
# end


@testset "ProductNode --- integration with Flux" begin
    d1 = 9
    d2 = 11
    nobs = 100
    x = randn(Float32, d1 + d2, nobs)

    m = ProductNode([SumProductSet.MvNormal(d1), SumProductSet.MvNormal(d2)])
    ps = Flux.params(m)

    @test !isempty(ps)
    @test !isnothing(gradient(() -> sum(SumProductSet.logpdf(m, x)), ps))
end

@testset "ProductNode --- integration with Mill" begin
    # Integration with Mill.ArrayNode
    d1 = 9
    d2 = 11
    nobs = 100
    x = Mill.ArrayNode(randn(Float32, d1 + d2, nobs))
    m = ProductNode([SumProductSet.MvNormal(d1), SumProductSet.MvNormal(d2)], [1:d1, d1+1:d1+d2])

    @test !isnothing(SumProductSet.logpdf(m, x))
    @test length(SumProductSet.logpdf(m, x)) == nobs
    @test SumProductSet.logpdf(m, x) ≈ SumProductSet.logpdf(m.components[1], x.data[1:d1, :]) + 
                                       SumProductSet.logpdf(m.components[2], x.data[d1+1:d1+d2, :])

    # Integration with Mill.ProductNode{<:NamedTuple}
    d1 = 9
    d2 = 11
    nobs = 100
    x = Mill.ProductNode(a=randn(Float32, d1, nobs), b=randn(Float32, d2, nobs))
    m = ProductNode(a=SumProductSet.MvNormal(d1), b=SumProductSet.MvNormal(d2))
    @test !isnothing(SumProductSet.logpdf(m, x))
    @test length(SumProductSet.logpdf(m, x)) == nobs
    @test SumProductSet.logpdf(m, x) ≈ SumProductSet.logpdf(m.components[1], x.data[:a]) + 
                                       SumProductSet.logpdf(m.components[2], x.data[:b])

end