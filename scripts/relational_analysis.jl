using DrWatson
@quickactivate
using Flux
using BSON
using Printf
using StatsBase
using DataFrames
using SumProductSet
using PGFPlotsX
using ColorSchemes
using PrettyTables
using EvalMetrics

# All the code depending on the neural networks (mlp, gru, lstm, hmil) will not work at this moment.

include("utils.jl")
# include("tree_mlp.jl")
# include("tree_gru.jl")
# include("tree_lstm.jl")


function axis(plots,
              xlabel::String="xlabel (-)",
              ylabel::String="ylabel (-)",
              width::String="122pt",
              height::String="122pt";
              title::String=" ",
              xmin=+0,
              xmax=+1,
              ymin=+0,
              ymax=+1,
              xtick=[0.0,0.2,0.4,0.6,0.8,1.0],
              ytick=[0.0,0.2,0.4,0.6,0.8,1.0],)
    title = replace(title, "_"=>" ")
    @pgf Axis({
        grid="both",
        xlabel=xlabel,
        ylabel=ylabel,
        width=width,
        height=height,
        legend_pos="south west",
        legend_cell_align="left",
        legend_style="{draw=none, fill=none, row sep=0.1pt}",
        title=title,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        # xtick=xtick,
        # ytick=ytick,
        # xticklabels=xtick,
        # yticklabels=ytick,
        # xmode="log",
        view=(0, 90),
        # x_label_style=raw"{{at={(0.5,-0.08)}}}",
        # y_label_style=raw"{{at={(-0.10,0.5)}}}",
    },
    plots)
end
function plot(x, y, color="red", mark="none", line_width=0.5, opacity=1.0)
    @pgf Plot({
        # dashed,
        thick,
        line_width=line_width,
        mark=mark,
        opacity=opacity,
        color=color
    },
    PGFPlotsX.Table(x=x, y=y))
end
function groupplot(plots,
                   n::Integer,
                   m::Integer,
                   width::String="150pt",
                   height::String="110pt")
    gp = @pgf GroupPlot({
            group_style =
            {
                group_size="$(m) by $(n)",
                horizontal_sep="46pt",
                vertical_sep="38pt",
            },
            grid="none",
            width=width,
            height=height,
            legend_pos="south west",
            legend_cell_align="left",
            legend_style="{draw=none, fill=none, row sep=0.1pt}",
            title_style=raw"{yshift=-5.0pt}",
            x_label_style=raw"{{at={(0.5,-0.12)}}}",
            y_label_style=raw"{{at={(-0.16,0.5)}}}",
        }
    )
    @pgf foreach(p->push!(gp, p), plots)
    gp
end


function convert_row(r)
    map(x->parse(Float32, x), getindex.(split.(collect(r), "\$\\pm\$"), 1))
end
function convert_element(e)
    if e isa String && contains(e, ".")
        return parse(Float32, getindex(split(e, "\$\\pm\$"), 1))
    else
        return nothing
    end
end
function rank_table(df::DataFrame, highlight::Array{<:NamedTuple,1}; latex::Bool=false)
    push!(df, ["rank", map(c->typeof(c)==String ? "" : 0f0, df[1, 2:end])...])
	foreach(highlight) do h
        m = map(eachrow(df[1:end-1, :])) do r
            StatsBase.competerank(convert_row(r[h.vs]), rev=h.rev)
        end
        df[end, h.vs] = map(x->@sprintf("%2.2f", x), mean(m))
	end

	if !latex
		h = map(highlight) do h
			h1 = Highlighter(f=(data, i, j)->i< size(df, 1)&&convert_element(data[i,j])==h.fa(convert_row(df[i, h.vs])), crayon=crayon"yellow bold")
			h2 = Highlighter(f=(data, i, j)->i==size(df, 1)&&convert_element(data[i,j])==h.fb(convert_row(df[i, h.vs])), crayon=crayon"yellow bold")
			h1, h2
		end
		pretty_table(df, highlighters=(collect(Iterators.flatten(h))...,), nosubheader=true)
	else
		h = map(highlight) do h
			h1 = LatexHighlighter((data, i, j)->i< size(df, 1)&&convert_element(data[i,j])==h.fa(convert_row(df[i, h.vs])), ["color{blue}","textbf"])
			h2 = LatexHighlighter((data, i, j)->i==size(df, 1)&&convert_element(data[i,j])==h.fb(convert_row(df[i, h.vs])), ["color{blue}","textbf"])
			h1, h2
		end
		pretty_table(df, backend=:latex, highlighters=(collect(Iterators.flatten(h))...,), nosubheader=true)
	end
end


predictive_probability(m::SumProductSet.AbstractModelNode, x) = softmax(SumProductSet.logjnt(m, x))
predictive_probability(m::ComposedFunction, x) = softmax(m(x))


function best_model(df, name, measure_find="lkl", measure_show="lkl", function_find=argmax, aggregation_type=:mean)
    m_find = Symbol("$(measure_find)_val")
    m_show = Symbol("$(measure_show)_tst")
    s_show = Symbol("$(measure_show)_s_tst")

    if (name == "mlp") | (name == "gru") | (name == "lstm" )
        df = groupby(df, [:dataset, :ni, :no, :bsize, :ssize])
    elseif name == "hmil"
        df = groupby(df, [:dataset,      :no, :bsize, :ssize])
    elseif name == "spsn"
        df = groupby(df, [:dataset, :pl, :ps, :bsize, :ssize])
    end

    if aggregation_type == :mean
        df = combine(df, m_find=>mean, m_show=>mean, m_show=>std=>s_show, AsTable(:m)=>x->vcat(x),
            AsTable(:seed_split)=>x->vcat(x), AsTable(:seed_init)=>x->vcat(x); renamecols=false)
        return combine(df->df[function_find(df[!, m_find]), :], groupby(df, :dataset))
    elseif aggregation_type == :maximum
        df = combine(df, m_find, m_show, :m, :seed_split, :seed_init; renamecols=false)
        return combine(df->df[function_find(df[!, m_find]), :], groupby(df, :dataset))
    end
end




function mea_tab_compute(; measure_find="lkl", measure_show="lkl", function_find=argmax, fa=maximum, fb=minimum, rev=true)
    df_mlp  = collect_results("data/tree_structures/accuracy"; rinclude=[r"mlp"])
    df_gru  = collect_results("data/tree_structures/accuracy"; rinclude=[r"gru"])
    df_lstm = collect_results("data/tree_structures/accuracy"; rinclude=[r"lstm"])
    df_hmil = collect_results("data/tree_structures/accuracy"; rinclude=[r"hmil"])
    df_spsn = collect_results("data/nodewise_spsn/accuracy")

    df_mlp  = best_model(df_mlp,  "mlp",  measure_find, measure_show, function_find)
    df_gru  = best_model(df_gru,  "gru",  measure_find, measure_show, function_find)
    df_lstm = best_model(df_lstm, "lstm", measure_find, measure_show, function_find)
    df_hmil = best_model(df_hmil, "hmil", measure_find, measure_show, function_find)
    df_spsn = best_model(df_spsn, "spsn", measure_find, measure_show, function_find)

    df_mlp  = mea_tab_compute(df_mlp,  measure_show)
    df_gru  = mea_tab_compute(df_gru,  measure_show)
    df_lstm = mea_tab_compute(df_lstm, measure_show)
    df_hmil = mea_tab_compute(df_hmil, measure_show)
    df_spsn = mea_tab_compute(df_spsn, measure_show)

    df_mlp  = mark_rows(df_mlp,  "mlp",  measure_show)
    df_gru  = mark_rows(df_gru,  "gru",  measure_show)
    df_lstm = mark_rows(df_lstm, "lstm", measure_show)
    df_hmil = mark_rows(df_hmil, "hmil", measure_show)
    df_spsn = mark_rows(df_spsn, "spsn", measure_show)

    df = innerjoin(df_mlp, df_gru, df_lstm, df_hmil, df_spsn; on=:dataset, makeunique=true)

    m_tst = Symbol.(filter(name->contains(name, "$(measure_show)_{tst}" ), names(df)))

    return rank_table(df, [(vs=m_tst, fa=fa, fb=fb, rev=rev)])
end
function mea_tab_compute(df, measure_show="acc")
    df = groupby(df, :dataset)

    r = map(pairs(df)) do (k, v)
        seed_split = v[!, :seed_split][1].seed_split
        m = v[!, :m][1].m

        data = read("$(dirdata)/$(k[1]).json", String)
        data = JSON3.read(data)
        x, y = data.x, data.y

        s = schema(x)
        e = suggestextractor(s)
        x = reduce(catobs, e.(x))

        o = map(seed_split, m) do s, m
            _, _, x_tst, _, _, y_tst = split_data(x, y, s)
            try
                p_tst = predictive_probability(m, x_tst)
                reliability(y_tst, p_tst, Val(10))[3]
            catch
                display(v)
                0f0
            end
        end
        mean(o), std(o)
    end
    DataFrame(dataset=map(k->k.dataset, keys(df)), ece_tst=getindex.(r, 1), ece_s_tst=getindex.(r, 1))
end

function mea_tab_collect(; measure_find="lkl", measure_show="lkl", function_find=argmax, fa=maximum, fb=minimum, rev=true)
    df_mlp  = collect_results("data/tree_structures/accuracy"; rinclude=[r"mlp"])
    df_gru  = collect_results("data/tree_structures/accuracy"; rinclude=[r"gru"])
    df_lstm = collect_results("data/tree_structures/accuracy"; rinclude=[r"lstm"])
    df_hmil = collect_results("data/tree_structures/accuracy"; rinclude=[r"hmil"])
    df_spsn = collect_results("data/nodewise_spsn/accuracy")

    df_mlp  = best_model(df_mlp,  "mlp",  measure_find, measure_show, function_find)
    df_gru  = best_model(df_gru,  "gru",  measure_find, measure_show, function_find)
    df_lstm = best_model(df_lstm, "lstm", measure_find, measure_show, function_find)
    df_hmil = best_model(df_hmil, "hmil", measure_find, measure_show, function_find)
    df_spsn = best_model(df_spsn, "spsn", measure_find, measure_show, function_find)

    df_mlp  = mark_rows(df_mlp,  "mlp",  measure_show)
    df_gru  = mark_rows(df_gru,  "gru",  measure_show)
    df_lstm = mark_rows(df_lstm, "lstm", measure_show)
    df_hmil = mark_rows(df_hmil, "hmil", measure_show)
    df_spsn = mark_rows(df_spsn, "spsn", measure_show)

    df = innerjoin(df_mlp, df_gru, df_lstm, df_hmil, df_spsn; on=:dataset, makeunique=true)

    m_tst = Symbol.(filter(name->contains(name, "$(measure_show)_{tst}" ), names(df)))

    return rank_table(df, [(vs=m_tst, fa=fa, fb=fb, rev=rev)])
end
function mark_rows(df, name, measure_show="lkl")
    m_show = Symbol("$(measure_show)_tst")
    s_show = Symbol("$(measure_show)_s_tst")
    combine(df, :dataset, [m_show, s_show]=>ByRow((m, s)->@sprintf("%2.2f\$\\pm\$%2.2f", m, s))=>"\$$(measure_show)_{tst}\$ "*name, renamecols=false)
end


function mis_fig_compute(name, measure_find="lkl", measure_show="lkl", function_find=argmax; r="")
    df_mlp  = collect_results("data/tree_structures/missing"; rinclude=[Regex("(dataset\\=$(r).+)(ctype\\=tree_mlp.+)")])
    df_gru  = collect_results("data/tree_structures/missing"; rinclude=[Regex("(dataset\\=$(r).+)(ctype\\=tree_gru.+)")])
    df_lstm = collect_results("data/tree_structures/missing"; rinclude=[Regex("(dataset\\=$(r).+)(ctype\\=tree_lstm.+)")])
    df_hmil = collect_results("data/tree_structures/missing"; rinclude=[Regex("(dataset\\=$(r).+)(ctype\\=hmil.+)")])
    df_spsn = collect_results("data/nodewise_spsn/missing"; rinclude=[Regex("(dataset\\=$(r).+)")])

    df_mlp  = best_model(df_mlp,  "mlp",  measure_find, measure_show, function_find)
    df_gru  = best_model(df_gru,  "gru",  measure_find, measure_show, function_find)
    df_lstm = best_model(df_lstm, "lstm", measure_find, measure_show, function_find)
    df_hmil = best_model(df_hmil, "hmil", measure_find, measure_show, function_find)
    df_spsn = best_model(df_spsn, "spsn", measure_find, measure_show, function_find)

    k, p_mlp  = mis_fig_compute(df_mlp,  "MLP",  colorschemes[:Dark2_5][1])
    _, p_gru  = mis_fig_compute(df_gru,  "GRU",  colorschemes[:Dark2_5][2])
    _, p_lstm = mis_fig_compute(df_lstm, "LSTM", colorschemes[:Dark2_5][4])
    _, p_hmil = mis_fig_compute(df_hmil, "HMIL", colorschemes[:Dark2_5][5])
    _, p_spsn = mis_fig_compute(df_spsn, "SPSN", colorschemes[:Dark2_5][3])

    p = hcat(p_mlp, p_gru, p_lstm, p_hmil, p_spsn)
    p = map((p, t)->axis(p, "proportion of missing values (-)", "accuracy (-)"; title=t, xmax=1.0), eachrow(p), k)

    # p = groupplot(vec(p), size(p, 2), size(p, 1))
    p = groupplot(vec(p), 5, 2)
    p = @pgf TikzPicture({font="\\scriptsize"}, p)
    pgfsave("mis_compute_$(name).tikz", p)
end
function mis_fig_compute(df::DataFrame, name, color)
    df = groupby(df, :dataset)

    mrate = [10e-4, 10e-2, 30e-2, 50e-2, 70e-2, 90e-2, 99e-2]

    p = map(pairs(df)) do (k, v)
        seed_split = v[!, :seed_split][1].seed_split
        m = v[!, :m][1].m

        data = read("$(dirdata)/$(k[1]).json", String)
        data = JSON3.read(data)
        x, y = data.x, data.y

        s = schema(x)
        e = suggestextractor(s)
        x = reduce(catobs, e.(x))

        o = mapreduce(hcat, mrate) do r
            map(seed_split, m) do s, m
                _, _, x_tst, _, _, y_tst = split_data(x, y, s)
                x_mis = make_missing(x_tst, r)
                mean(Flux.onecold(predictive_probability(m, x_mis)) .== y_tst)
            end
        end

        acc = vec(mean(o; dims=1))

        vcat(plot(mrate, acc, color, 0.5, 1.0), LegendEntry("$(name)"))
    end
    k = map(k->k[1], keys(df))

    k, p
end

function mis_fig_collect(name, measure_show="acc", measure_type="tst")
    df_mlp  = collect_results("data/tree_structures/missing"; rinclude=[r"mlp"])
    df_gru  = collect_results("data/tree_structures/missing"; rinclude=[r"gru"])
    df_lstm = collect_results("data/tree_structures/missing"; rinclude=[r"lstm"])
    # df_hmil = collect_results("data/tree_structures/missing"; rinclude=[r"hmil"])
    df_spsn = collect_results("data/nodewise_spsn/missing")

    k, p_mlp  = mis_fig_collect(df_mlp,  measure_show, measure_type, colorschemes[:Dark2_5][1], "mlp")
    _, p_gru  = mis_fig_collect(df_gru,  measure_show, measure_type, colorschemes[:Dark2_5][2], "gru")
    _, p_lstm = mis_fig_collect(df_lstm, measure_show, measure_type, colorschemes[:Dark2_5][4], "lstm")
    # _, p_hmil = mis_fig_collect(df_hmil, measure_show, measure_type, colorschemes[:Dark2_5][5], "hmil")
    _, p_spsn = mis_fig_collect(df_spsn, measure_show, measure_type, colorschemes[:Dark2_5][3], "spsn")

    p = hcat(p_mlp, p_gru, p_lstm, p_spsn)
    p = map((p, t)->axis(p, "missing rate (-)", "accuracy (-)"; title=t, xtick=[0.0,0.2,0.4,0.6,0.8,1.0], ytick=[0.0,0.2,0.4,0.6,0.8,1.0], xmax=1.0), eachrow(p), k)

    p = groupplot(vec(p), size(p, 2), size(p, 1))
    p = @pgf TikzPicture({font="\\scriptsize"}, p)
    pgfsave("missing_$(name).pdf", p)
end
function mis_fig_collect(di, measure_show="acc", measure_type="tst", color="blue", name="")
    m_show = Symbol("$(measure_show)_$(measure_type)")

    df = groupby(di, [:dataset, :mrate])
    df = combine(df, m_show=>mean, renamecols=false)
    df = groupby(df, :dataset)

    p = map(pairs(df)) do (k, v)
        vcat(plot(v[:, :mrate], v[:, m_show], color, 0.5), LegendEntry(name))
    end
    k = map(k->k[1], keys(df))

    k, p
end


mea_tab_collect(; measure_find="acc", measure_show="acc", function_find=argmax, fa=maximum, fb=minimum, rev=true)
# mea_tab_collect(; measure_find="acc", measure_show="ece", function_find=argmax, fa=minimum, fb=minimum, rev=false)
# mea_tab_compute(; measure_find="acc", measure_show="acc", function_find=argmax, fa=maximum, fb=minimum, rev=true)
# mea_tab_compute(; measure_find="acc", measure_show="ece", function_find=argmax, fa=minimum, fb=minimum, rev=false)

# mis_fig_compute("all", "acc", "acc") #; r="(cora|genes?)"
# mis_fig_collect("all", "acc", "tst")
