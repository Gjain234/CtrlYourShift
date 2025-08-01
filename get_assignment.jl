# get_assignment.jl

import Pkg

required_packages = [
    "CSV", "DataFrames", "JuMP", "Gurobi",
    "Statistics", "StatsBase"
]

for pkg in required_packages
    try
        @eval import $(Symbol(pkg))
    catch
        println("Installing missing package: $pkg")
        Pkg.add(pkg)
    end
end

# Now load the packages
using CSV, DataFrames, JuMP, Gurobi, Statistics, StatsBase

# ---- Parse Arguments ----
if length(ARGS) < 4
    error("Usage: julia get_assignment.jl <dataset> <grouping_var> <outcome_var> <model_method>")
end

dataset = ARGS[1]
group = ARGS[2]
outcome = ARGS[3]
model_method = ARGS[4]  # e.g., "xlearn_bart"

split_vals = split(model_method, "_")
model = split_vals[end]
method = join(split_vals[1:end-1], "_")

base_path = "results/model_comparison/$(dataset)/"
assign_out_path = joinpath("results", "reward", dataset)
if !isdir(assign_out_path)
    mkpath(assign_out_path)
end

assign_file = joinpath(base_path, "$(method)_$(model)_$(dataset)_shifted_all_predictions.csv")
global_pred_file = joinpath(base_path, "global_$(model)_$(dataset)_shifted_predictions.csv")
assign_preds = Matrix{Float64}(CSV.read(assign_file, DataFrame))
df = CSV.read(global_pred_file, DataFrame)

all_groups = String.(df.group)
group_names = sort(unique(all_groups))
y = Vector{Float64}(df.ground_truth)
n = length(all_groups)
n_groups = length(group_names)
group_counts = countmap(all_groups)
group_sizes = [group_counts[g] for g in group_names]

function get_assignment(predictions::Matrix{Float64}, capacities::Union{Vector{Int}, Nothing}, group_names::Vector{String})
    n, n_groups = size(predictions)
    if isnothing(capacities)
        best_indices = [argmax(predictions[i, :]) for i in 1:n]
        return group_names[best_indices]
    else
        model = Model(Gurobi.Optimizer)
        set_silent(model)
        @variable(model, 0 <= X[1:n, 1:n_groups] <= 1)
        @objective(model, Max, sum(X .* predictions))
        @constraint(model, [i=1:n], sum(X[i, :]) == 1)
        @constraint(model, [j=1:n_groups], sum(X[:, j]) == capacities[j])
        optimize!(model)
        X_val = value.(X)
        assigned_indices = [argmax(X_val[i, :]) for i in 1:n]
        return group_names[assigned_indices]
    end
end

for constraint in ["Capacity", "Unconstrained"]
    cap = constraint == "Capacity" ? group_sizes : nothing
    assignment = get_assignment(assign_preds, cap, group_names)
    assign_out_file = joinpath(assign_out_path, "assignment_$(method)_$(model)_$(constraint).csv")
    CSV.write(assign_out_file, DataFrame(Assignment=assignment))
end
println("Assignments saved!")