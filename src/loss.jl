
function ul_loss(m::SumNode, xu)
    # E-step for unlabeled data
    p = []
    Flux.Zygote.ignore() do 
        p = softmax(logjnt(m, xu); dims=1)
    end
    -mean(p .* logjnt(m, xu))
end 

# inefficient implementation of supervised loss
function sl_loss(m::SumNode, xl, yl::Vector{Int})
    -mean(logjnt(m, xl)[CartesianIndex.(yl, 1:length(yl))])
end

function ssl_loss(m::SumNode, xu, xl, yl::Vector{Int})
    nu = nobs(xu) 
    nl = nobs(xl)

    (nl*sl_loss(m, xl, yl) + nu*ul_loss(m, xu))/(nu+nl)
end 
