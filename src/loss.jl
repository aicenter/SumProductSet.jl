
function em_loss(m::SumNode, xu)
    # E-step for unlabeled data
    p = []
    Flux.Zygote.ignore() do 
        p = softmax(logjnt(m, xu); dims=1)
    end
    # M-step objective
    -mean(p .* logjnt(m, xu))
end 

ce_loss(m::SumNode, xl, yl::Vector{Int}) = -mean(logjnt(m, xl)[CartesianIndex.(yl, 1:length(yl))])

function ssl_loss(m::SumNode, xu, xl, yl::Vector{Int})
    nu = nobs(xu) 
    nl = nobs(xl)

    (nl*ce_loss(m, xl, yl) + nu*em_loss(m, xu))/(nu+nl)
end 
