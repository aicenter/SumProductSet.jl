
function rank(m::SetNode, x::Mill.BagNode)
    bags = x.bags.bags
    logp_f = logpdf(m.feature, x.data)
    bags_cards = hcat(length.(bags)...)
    rb = SumProductSet.logpdf(m.cardinality, bags_cards)

    @inbounds for (bi, b) in enumerate(bags)
        for i in b
            rb[bi] += logp_f[i] #/ bags_cards[bi]
        end
        rb[bi] += logfactorial(bags_cards[bi])
    end
    rb
end

rankjnt(m::SumNode, x) = mapreduce(c->rank(c, x), vcat, m.components) .+ hcat(logsoftmax(m.weights))
rank(m::SumNode, x) = logsumexp(rankjnt(m, x), dims=1)

rank(m, x) = logpdf(m, x)