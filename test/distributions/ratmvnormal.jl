
@testset "DiagRMvNormal --- rrule test" begin
    ndims = 10
    nobs = 100
    dtype = Float64
    m = SumProductSet.DiagRMvNormal(ndims; dtype=dtype)
    x = rand(m, nobs)
    s = SumProductSet.σs(m.logs, m.min_s, m.max_s)

    test_rrule(SumProductSet._logpdf_grdiag, m.m, s, x.data ⊢ NoTangent();
                check_inferred=true, rtol = 1.0e-9, atol = 1.0e-9)
end

@testset "IsoRMvNormal --- rrule test" begin
    ndims = 10
    nobs = 100
    dtype = Float64
    m = SumProductSet.IsoRMvNormal(rand(dtype, ndims), rand(dtype, 1))
    x = rand(m, nobs)
    s = SumProductSet.σs(m.logs, m.min_s, m.max_s)

    test_rrule(SumProductSet._logpdf_griso, m.m, s, x.data ⊢ NoTangent();
                check_inferred=true, rtol = 1.0e-9, atol = 1.0e-9)
end

@testset "UnitMvNormal --- rrule test" begin
    ndims = 10
    nobs = 100
    dtype = Float64
    m = SumProductSet.UnitMvNormal(rand(dtype, ndims))
    x = rand(m, nobs)

    test_rrule(SumProductSet._logpdf_gunit, m.m, x.data ⊢ NoTangent();
                check_inferred=true, rtol = 1.0e-9, atol = 1.0e-9)
end

