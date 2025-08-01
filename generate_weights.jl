# Install necessary packages if not already installed
import Pkg
Pkg.add(["JuMP", "Gurobi", "StatsBase", "DataFrames", "Random", "CSV","LinearAlgebra"])
using JuMP, Gurobi, StatsBase, DataFrames, Random, CSV, LinearAlgebra

# --- Read in arguments ---
if length(ARGS) < 5
    error("Not enough arguments. Usage: julia generate_weights.jl model data_type grouping_var shifted iter")
end

model = ARGS[1]
data_type = ARGS[2]
grouping_var = ARGS[3]
shifted = parse(Bool, ARGS[4])
iter = parse(Int, ARGS[5])

println("Running: model=$model, data_type=$data_type, grouping_var=$grouping_var, shifted=$shifted, iter=$iter")

# Construct file paths
file_base = "$(model)_$(data_type)"
file_base_with_iter = "$(file_base)_$(iter)"
file_base_with_group = "$(file_base)_$(grouping_var)"

# Create output directory
output_dir = "checkpoints/weights/$(file_base_with_group)$(shifted ? "_shifted" : "")"
mkpath(output_dir)
output_path = joinpath(output_dir, "clustered_weights_seed_$(iter).csv")

# Check if output already exists
if isfile(output_path)
    println("Output already exists at $output_path — skipping iteration $iter.")
    exit()
end

# Add checkpoint path
checkpoint_path = joinpath(output_dir, "checkpoint_seed_$(iter).csv")

# --- Load data ---
global_pred_file = "predictions/test_global_$(file_base_with_iter).csv"
backtest_info_file = "data/backtest_info_$(file_base_with_group)_$(iter).csv"
residual_pred_file = "predictions/test_residual_$(file_base_with_group)$(shifted ? "_shifted" : "")_$(iter).csv"

global_pred_df = select(CSV.read(global_pred_file, DataFrame), "x")
backtest_cohort_df = CSV.read(backtest_info_file, DataFrame)
residual_pred_df = CSV.read(residual_pred_file, DataFrame)

# --- Setup ---
full_M = size(residual_pred_df, 2)
M=5
group_column = backtest_cohort_df[:, grouping_var]
group_counts = countmap(group_column)
sorted_loc_names = sort(collect(keys(group_counts)))
n_locations = length(sorted_loc_names)
results_matrix = zeros(n_locations, full_M)

# Check for checkpoint
start_loc_idx = 1
if isfile(checkpoint_path)
    println("Checkpoint found at $checkpoint_path. Resuming from last saved location.")
    checkpoint_df = CSV.read(checkpoint_path, DataFrame)
    results_matrix[1:size(checkpoint_df, 1), :] .= Matrix(checkpoint_df[:, 1:end])
    start_loc_idx = size(checkpoint_df, 1) + 1
end

# --- Complex Optimization-based weights ---
function solve_NLMIP(idx, M, nA, y, p, nj)
    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "NonConvex", 2)

    @variable(model, z[1:M], Bin)
    @variable(model, θ[1:nA])
    θ_min = -1.0
    θ_max = 1.0

    for i in 1:nA
        @constraint(model, θ[i] >= θ_min)
        @constraint(model, θ[i] <= θ_max)
        @constraint(model, sum(p[i,j] * nj[j] * z[j] for j in 1:M) ==
                             sum(nj[j] * z[j] for j in 1:M) * θ[i])
    end

    @constraint(model, z[idx] == 1)
    @objective(model, Min, sum((y[i] - θ[i])^2 for i in 1:nA))
    optimize!(model)

    return value.(z)
end


for loc_idx in start_loc_idx:n_locations
    current_loc = sorted_loc_names[loc_idx]
    println("Solving for location $current_loc")
    all_other_idxs = setdiff(1:n_locations, [loc_idx])
    random_sample_idxs = sort(all_other_idxs[randperm(length(all_other_idxs))[1:M-1]])
    selected_idxs = sort([loc_idx; random_sample_idxs])
    nA = group_counts[current_loc]

    idxs = backtest_cohort_df[:, grouping_var] .== current_loc

    y = backtest_cohort_df[idxs, 2] .-
        global_pred_df[idxs,1]

    p = residual_pred_df[idxs, selected_idxs]
    y .-= mean(y)


    nj_subset = Dict(j => group_counts[sorted_loc_names[selected_idxs[j]]] for j in 1:M)
    local_idx = findfirst(x -> x == loc_idx, selected_idxs)

    z_values = solve_NLMIP(local_idx, M, nA, y, p, nj_subset)

    for j in 1:M
        global_idx = selected_idxs[j]
        results_matrix[loc_idx, global_idx] = z_values[j]
    end

    # Save checkpoint after each location
    checkpoint_df = DataFrame(results_matrix[1:loc_idx, :], Symbol.(sorted_loc_names))
    CSV.write(checkpoint_path, checkpoint_df)
end

# --- Save Results ---
results_matrix = DataFrame(results_matrix, Symbol.(sorted_loc_names))
CSV.write(output_path, results_matrix)

# Remove checkpoint file after successful completion
if isfile(checkpoint_path)
    rm(checkpoint_path; force=true)
end

