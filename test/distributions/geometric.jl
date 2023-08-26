
@testset "Geometric --- logpdf forward" begin
	ndims = 10
    nobs = 100
    m = SumProductSet.Geometric(ndims)
    x1 = rand(0:50, ndims, nobs) |> SparseMatrixCSC

    @test length(SumProductSet.logpdf(m, x1)) == nobs
end

@testset "Geometric --- rand sampling" begin
    ndims = 10
    nobs = 100
    m = SumProductSet.Geometric(ndims)
    @test typeof(rand(m)) <: Mill.ArrayNode
    @test typeof(rand(m).data) <: SparseMatrixCSC
    @test size(rand(m)) == (ndims, 1)

    @test typeof(rand(m, nobs)) <: Mill.ArrayNode
    @test typeof(rand(m, nobs).data) <: SparseMatrixCSC
    @test size(rand(m, nobs)) == (ndims, nobs)
end

@testset "Geometric --- rrule test" begin
    ndims = 10
    nobs = 100
    dtype = Float64
	m = SumProductSet.Geometric(ndims; dtype=dtype)
    x = rand(m, nobs)

    test_rrule(SumProductSet._logpdf_geometric, m.logitp, x.data ⊢ NoTangent();
                check_inferred=true, rtol = 1.0e-9, atol = 1.0e-9)
end

@testset "Geometric --- integration with Flux" begin
    ndims = 10
    nobs = 100
	m = SumProductSet.Geometric(ndims)
	ps = Flux.params(m)

    @test !isempty(ps)
    x = rand(0:25, ndims, nobs) |> SparseMatrixCSC
    @test !isnothing(gradient(() -> sum(SumProductSet.logpdf(m, x)), ps))
end


@testset "Geometric --- correctness" begin
    p = 0.7
    logitp = log(p/(1-p))
    n = 100
    m1 = Distributions.Geometric(p)
    m2 = SumProductSet.Geometric([logitp])
    x1 = rand(m1, n) 
    x2 = rand(m2, n)

    @test Distributions.logpdf.(m1, x1)[:] ≈ 
            SumProductSet.logpdf(m2, hcat(x1...) |> SparseMatrixCSC)[:]
    @test Distributions.logpdf.(m1, x2.data |> Matrix)[:] ≈ 
            SumProductSet.logpdf(m2, x2)[:]


    p1, p2 = 0.7, 0.1
    logitp1, logitp2 = log(p1/(1-p1)), log(p2/(1-p2))
    m3 = SumProductSet.Geometric([logitp1, logitp2])
    m31 = SumProductSet.Geometric([logitp1])
    m32 = SumProductSet.Geometric([logitp2])

    x3 = rand(m3, n)
    @test SumProductSet.logpdf(m31, x3.data[1:1, :]) + SumProductSet.logpdf(m32, x3.data[2:2, :]) ≈
            SumProductSet.logpdf(m3, x3)

end
