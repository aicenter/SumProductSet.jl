
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

