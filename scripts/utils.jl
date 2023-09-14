using Flux
using Printf
using Statistics
using EvalMetrics

include("utils_hmil.jl")
include("utils_eval.jl")
include("utils_data.jl")


function logjnt(m::SumNode, x::Union{AbstractMatrix, Mill.AbstractMillNode}, y)
    l = logsoftmax(mapreduce(c->logpdf(c, x), vcat, m.components) .+ logsoftmax(m.weights))
    l[CartesianIndex.(y, 1:length(y))]
end


Base.length(x::Mill.ProductNode) = Mill.nobs(x)
predict(m, x) = mapslices(argmax, softmax(SumProductSet.logjnt(m, x)), dims=1)[:]


function evaluate(m, x_trn::Ar, x_val::Ar, x_tst::Ar, y_trn::Ai, y_val::Ai, y_tst::Ai) where {Ar<:Mill.AbstractMillNode,Ai<:Vector{<:Int}}
    lkl_trn = mean(logpdf(m, x_trn))
    lkl_val = mean(logpdf(m, x_val))
    lkl_tst = mean(logpdf(m, x_tst))

    acc_trn = mean(y_trn .== predict(m, x_trn))
    acc_val = mean(y_val .== predict(m, x_val))
    acc_tst = mean(y_tst .== predict(m, x_tst))

    acc_bin_trn, con_bin_trn, ece_trn, his_trn = reliability(y_trn, softmax(SumProductSet.logjnt(m, x_trn)), Val(10))
    acc_bin_val, con_bin_val, ece_val, his_val = reliability(y_val, softmax(SumProductSet.logjnt(m, x_val)), Val(10))
    acc_bin_tst, con_bin_tst, ece_tst, his_tst = reliability(y_tst, softmax(SumProductSet.logjnt(m, x_tst)), Val(10))

    (; lkl_trn,     lkl_val,     lkl_tst,
       acc_trn,     acc_val,     acc_tst,
       ece_trn,     ece_val,     ece_tst,
       his_trn,     his_val,     his_tst,
       acc_bin_trn, acc_bin_val, acc_bin_tst,
       con_bin_trn, con_bin_val, con_bin_tst)
end

function evaluate_ad(m, x_trn::Ar, x_val::Ar, x_tst::Ar, y_val::Ai, y_tst::Ai) where {Ar<:Mill.AbstractMillNode,Ai<:Vector{<:Int}}
    lkl_trn = mean(logpdf(m, x_trn))
    lkl_val = mean(logpdf(m, x_val))
    lkl_tst = mean(logpdf(m, x_tst))

    # -rank to get anomaly score
    auc_val = binary_eval_report(y_val, -logpdf(m, x_val)[:])["au_roccurve"]
    auc_tst = binary_eval_report(y_tst, -logpdf(m, x_tst)[:])["au_roccurve"]

    (; lkl_trn,     lkl_val,     lkl_tst,
                    auc_val,     auc_tst)
end

function gd!(m, x_trn::Ar, x_val::Ar, x_tst::Ar,
                y_trn::Ai, y_val::Ai, y_tst::Ai,
                o, nepoc::Int, bsize::Int, config_exp, config_wat, folder; p::Flux.Params=Flux.params(m), ftype::Type=Float32) where {Ar<:Mill.AbstractMillNode,Ai<:AbstractArray{<:Int,1}}
    t_trn, l_trn, l_val, l_tst, a_trn, a_val, a_tst = ftype[], ftype[], ftype[], ftype[], ftype[], ftype[], ftype[]
    d_trn = Flux.DataLoader((x_trn, y_trn); batchsize=bsize)
    final = :maximum_iterations
    o_trn = -ftype(Inf)
    o_val = +ftype(0e0)

    for e in 1:nepoc
        t̄_trn = @elapsed begin
            for (x_trn, y_trn) in d_trn # Flux.DataLoader((x_trn, y_trn); batchsize=bsize, shuffle=true)
                g = gradient(()->-mean(logjnt(m, x_trn, y_trn)), p)
                Flux.Optimise.update!(o, p, g)
            end
        end

        eval = evaluate(m, x_trn, x_val, x_tst, y_trn, y_val, y_tst)

        l_dif = eval.lkl_trn - o_trn
        o_trn = eval.lkl_trn

        abs(l_dif) <= ftype(1e-8) && (final = :absolute_tolerance; break)
        isnan(eval.lkl_trn)       && (final = :nan;                break)

        if mod(e, 2) == 1
            push!(t_trn, t̄_trn)
            push!(l_trn, eval.lkl_trn)
            push!(l_val, eval.lkl_val)
            push!(l_tst, eval.lkl_tst)
            push!(a_trn, eval.acc_trn)
            push!(a_val, eval.acc_val)
            push!(a_tst, eval.acc_tst)
        end

        @printf("gd: epoch: %i | l_trn %2.2f | l_val %2.2f | l_tst %2.2f || a_trn %2.2f | a_val %2.2f | a_tst %2.2f |\n",
            e, eval.lkl_trn, eval.lkl_val, eval.lkl_tst, eval.acc_trn, eval.acc_val, eval.acc_tst)

        if eval.acc_val > o_val
            npars = length(Flux.destructure(m)[1])
            produce_or_load(datadir("nodewise_spsn/$(folder)"), config_exp; config_wat...) do config
                ntuple2dict(merge(config, eval, (; t_trn, l_trn, l_val, l_tst, a_trn, a_val, a_tst, final), (; m, npars)))
            end
            o_val = eval.acc_val
        end
    end
end

function gd_ad!(m, x_trn::Ar, x_val::Ar, x_tst::Ar,
                              y_val::Ai, y_tst::Ai,
                o, nepoc::Int, bsize::Int, config_exp, config_wat, folder; p::Flux.Params=Flux.params(m), ftype::Type=Float32) where {Ar<:Mill.AbstractMillNode,Ai<:AbstractArray{<:Int,1}}
    t_trn, l_trn, l_val, l_tst, auc_trn, auc_val, auc_tst = ftype[], ftype[], ftype[], ftype[], ftype[], ftype[], ftype[]
    d_trn = Flux.DataLoader(x_trn; batchsize=bsize)
    final = :maximum_iterations
    o_trn = -ftype(Inf)
    o_val = +ftype(0e0)

    for e in 1:nepoc
        t̄_trn = @elapsed begin
            for x_trn in d_trn # Flux.DataLoader((x_trn, y_trn); batchsize=bsize, shuffle=true)
                g = gradient(()->-mean(logpdf(m, x_trn)), p)
                Flux.Optimise.update!(o, p, g)
            end
        end

        eval = evaluate_ad(m, x_trn, x_val, x_tst, y_val, y_tst)

        l_dif = eval.lkl_trn - o_trn
        o_trn = eval.lkl_trn

        abs(l_dif) <= ftype(1e-8) && (final = :absolute_tolerance; break)
        isnan(eval.lkl_trn)       && (final = :nan;                break)

        if mod(e, 2) == 1
            push!(t_trn, t̄_trn)
            push!(l_trn, eval.lkl_trn)
            push!(l_val, eval.lkl_val)
            push!(l_tst, eval.lkl_tst)
            push!(auc_val, eval.auc_val)
            push!(auc_tst, eval.auc_tst)
        end

        @printf("gd: epoch: %i | l_trn %2.2f | l_val %2.2f | l_tst %2.2f || auc_val %2.2f | auc_tst %2.2f |\n",
            e, eval.lkl_trn, eval.lkl_val, eval.lkl_tst, eval.auc_val, eval.auc_tst)

        if eval.auc_val > o_val
            npars = length(Flux.destructure(m)[1])
            produce_or_load(datadir("nodewise_spsn/$(folder)"), config_exp; config_wat...) do config
                ntuple2dict(merge(config, eval, (; t_trn, l_trn, l_val, l_tst, auc_trn, auc_val, auc_tst, final), (; m, npars)))
            end
            o_val = eval.auc_val
        end
    end
end

