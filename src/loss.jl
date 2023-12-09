
function em_loss(m::SumNode, xu)
    # E-step for unlabeled data
    p = []
    Flux.Zygote.ignore() do 
        p = softmax(logjnt(m, xu); dims=1)
    end
    # M-step objective
    -mean(p .* logjnt(m, xu))
end 


# disc loss is actually a cross entropy loss  

disc_loss(m::SumNode, xl, yl::Vector{Int}) = -mean(logsoftmax(logjnt(m, xl), dims=1)[CartesianIndex.(yl, 1:length(yl))])
gen_loss(m::SumNode, xl, yl::Vector{Int}) = -mean(logjnt(m, xl)[CartesianIndex.(yl, 1:length(yl))])
