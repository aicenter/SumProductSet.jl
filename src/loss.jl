
"""
    em_loss(m::SumNode, xu)
Expetation maximization algorithm objective (multiplied by the minus sign) `for the top SumNode` layer for unlabeled data `xu``.  
"""
function em_loss(m::SumNode, xu)
    # E-step for unlabeled data
    p = []
    Flux.Zygote.ignore() do 
        p = softmax(logjnt(m, xu); dims=1)
    end
    # minus M-step objective
    -mean(p .* logjnt(m, xu))
end 

"""
    ce_loss(m::SumNode, xl, yl::Vector{Int})
Cross entropy loss / negative log-likelihood loss for labeled data `xl` with corresponding labels `yl`. 
"""
ce_loss(m::SumNode, xl, yl::Vector{Int}) = -mean(logjnt(m, xl)[CartesianIndex.(yl, 1:length(yl))])
