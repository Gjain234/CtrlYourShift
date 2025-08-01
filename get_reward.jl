import Pkg

required_packages = [
    "CSV", "DataFrames", "Statistics", "StatsBase",
    "CategoricalArrays", "StatsPlots"
]

for pkg in required_packages
    try
        @eval import $(Symbol(pkg))
    catch
        println("Installing missing package: $pkg")
        Pkg.add(pkg)
    end
end

# Now bring them into scope
using CSV, DataFrames, Statistics, StatsBase, CategoricalArrays, StatsPlots

# ---- Parse Arguments ----
if length(ARGS) < 3
    error("Usage: julia get_reward.jl <dataset> <grouping_var> <outcome_var>")
end

dataset = ARGS[1]
group = ARGS[2]
outcome = ARGS[3]

architectures = ["regression", "tree", "ranger", "bart"]
methods = ["global", "xlearn", "local", "grouped_xlearn"]
method_order = ["local", "global", "xlearn", "grouped_xlearn"]
color_palette = [:blue, :orange, :green, :purple]

base_path = "results/model_comparison/$(dataset)/"
assign_out_path = joinpath("results", "reward", dataset)
results_out_file = joinpath(assign_out_path, "reward.csv")

if !isdir(assign_out_path)
    mkpath(assign_out_path)
end

# ---- Load All BART + xlearn Models for Reward Eval ----
all_16_models = Dict{String, Matrix{Float64}}()
for arch in ["bart"], method in ["xlearn"]
    pred_file = joinpath(base_path, "$(method)_$(arch)_$(dataset)_shifted_all_predictions.csv")
    if isfile(pred_file)
        df = CSV.read(pred_file, DataFrame)
        all_16_models["$(method)_$(arch)"] = Matrix{Float64}(df)
    else
        @warn "Missing prediction file: $pred_file"
    end
end

# ---- Helper Function: Unbiased Reward ----
function compute_unbiased_reward(assignment::Vector{String}, group_names::Vector{String},
                                  y::Vector{Float64}, all_groups::Vector{String},
                                  model_list::Dict{String, Matrix{Float64}},
                                  target_groups::Union{Nothing, Vector{String}}=nothing)
    n = length(assignment)
    per_model_rewards = Float64[]
    se = 0
    for (_, model_mat) in model_list
        reward = 0.0
        all_rewards = Float64[]
        for (l, g) in enumerate(group_names)
            if !isnothing(target_groups) && !(g in target_groups)
                continue
            end
            C_l = findall(x -> x == g, assignment)
            A_l = findall(x -> x == g, all_groups)
            C_int_A = intersect(C_l, A_l)

            aug_term = isempty(C_int_A) ? 0.0 :
                (length(C_l) / length(C_int_A)) * sum(y[C_int_A] .- model_mat[C_int_A, l])
            model_term = sum(model_mat[C_l, l])
            push!(all_rewards, aug_term + model_term)
            reward += aug_term + model_term
        end
        se = std(all_rewards) / sqrt(n)
        push!(per_model_rewards, reward / n)
    end
    return (mean=mean(per_model_rewards), sd=se)
end

# ---- Loop Over Models and Methods ----
results = DataFrame[]

for model in architectures
    global_pred_file = joinpath(base_path, "global_$(model)_$(dataset)_shifted_predictions.csv")
    if !isfile(global_pred_file)
        @warn "Skipping model=$model due to missing global predictions file."
        continue
    end

    df = CSV.read(global_pred_file, DataFrame)
    all_groups = String.(df.group)
    y = Vector{Float64}(df.ground_truth)
    group_names = sort(unique(all_groups))

    group_counts = countmap(all_groups)
    group_sizes = [group_counts[g] for g in group_names]
    group_counts = sort(countmap(all_groups); byvalue=true)
    cutoff = ceil(Int, length(group_counts) / 3)
    smallest_groups = collect(keys(group_counts))[1:cutoff]

    for method in methods
        for constraint in ["Capacity", "Unconstrained"]
            assign_file = joinpath(assign_out_path, "assignment_$(method)_$(model)_$(constraint).csv")
            if !isfile(assign_file)
                @warn "Missing assignment file: $assign_file"
                continue
            end
            assignment = String.(CSV.read(assign_file, DataFrame).Assignment)

            reward_full = compute_unbiased_reward(assignment, group_names, y, all_groups, all_16_models)
            reward_small = compute_unbiased_reward(assignment, group_names, y, all_groups, all_16_models, smallest_groups)

            push!(results, DataFrame(Model=model, Assignment=method, Constraint=constraint,
                                     RewardType="Full", RewardPerUnit=reward_full.mean, SD=reward_full.sd))
            push!(results, DataFrame(Model=model, Assignment=method, Constraint=constraint,
                                     RewardType="SmallGroups", RewardPerUnit=reward_small.mean, SD=reward_small.sd))
        end
    end
end

# ---- Save and Plot Results ----
results_df = vcat(results...)
results_df.Model = categorical(results_df.Model, ordered=true, levels=architectures)
results_df.Assignment = categorical(results_df.Assignment, ordered=true, levels=methods)
results_df.Constraint = categorical(results_df.Constraint, ordered=true, levels=["Capacity", "Unconstrained"])
results_df.RewardType = categorical(results_df.RewardType, ordered=true, levels=["Full", "SmallGroups"])
CSV.write(results_out_file, results_df)

# ---- Plotting ----
results_df.Assignment = categorical(results_df.Assignment, ordered=true, levels=method_order)
plot_list = []
n_methods = length(method_order)
bar_width = 0.8 / n_methods

for reward_type in levels(results_df.RewardType), constraint in levels(results_df.Constraint)
    subdf = filter(row -> row.RewardType == reward_type && row.Constraint == constraint, results_df)
    x_categories = levels(results_df.Model)
    x_pos = 1:length(x_categories)

    plt = plot(legend = false,
               xticks = (x_pos, x_categories),
               xlabel = "Model Architecture",
               ylabel = "Reward per Unit",
               title = "$(reward_type) Reward — $(constraint)")

    for (i, method) in enumerate(method_order)
        method_df = filter(row -> row.Assignment == method, subdf)
        y_vals = fill(NaN, length(x_categories))
        y_errs = fill(NaN, length(x_categories))
        for (j, model) in enumerate(x_categories)
            row = filter(r -> r.Model == model, method_df)
            if nrow(row) > 0
                y_vals[j] = row.RewardPerUnit[1]
                y_errs[j] = row.SD[1]
            end
        end

        shifted_x = collect(x_pos) .- 0.4 .+ bar_width * (i - 0.5)

        bar!(shifted_x, y_vals;
             yerror = y_errs,
             bar_width = bar_width,
             label = method,
             color = color_palette[i])
    end

    push!(plot_list, plt)
end

legend_plot = plot(legend = :bottom, size = (800, 100),
                   framestyle = :none, xaxis = false, yaxis = false)
for (i, m) in enumerate(method_order)
    plot!(legend_plot, [NaN], label = m, color = color_palette[i])
end

final_plot = plot(plot_list..., layout = (2, 2), size = (1000, 1000))
final_with_legend = plot(final_plot, legend_plot, layout = @layout([a{0.9h}; b{0.1h}]))

savefig(final_with_legend, "plots/$(dataset)/reward_plots.pdf")
println("Reward plots saved to plots/$(dataset)/reward_plots.pdf")