using Statistics


function reliability(y::Vector{<:Int}, p::Matrix{T}, ::Val{M}) where {T<:Real,M}
    o = mapslices(findmax, p, dims=1)
    p̂ = getindex.(o, 1)
    ŷ = getindex.(o, 2)
    n = length(y)

    bin = map(m->((m-1)/M, m/M), 1:M)
    idx = map(bin) do b
        mapreduce(vcat, enumerate(p̂)) do (j, p)
            ((b[1] <= p) && (p <= b[2])) ? j : []
        end
    end

    acc = map(i->length(i) > 0 ? mean(ŷ[i] .== y[i]) : T(0e0), idx)
    con = map(i->length(i) > 0 ? mean(p̂[i])          : T(0e0), idx)
    his = map(length, idx)

    ece = mapreduce((i, a, c)->(length(i)/n)*abs(a-c), +, idx, acc, con)

    acc, con, ece, his
end

entropy(p::Matrix{T}; dims=1) where {T<:Real} = -sum(p.*log.(p); dims=dims)[:]

function outlier_entropy(p::Matrix{T}, ::Val{V}, ::Val{M}) where {T<:Real,V,M}
    e = entropy(p)
    n = size(p, 2)

    thr = V*collect(1:M) / M
    idx = map(thr) do t
        mapreduce(vcat, enumerate(e)) do (j, e)
            e >= t ? j : []
        end
    end

    his = map(length, idx) / n

    thr, his
end
