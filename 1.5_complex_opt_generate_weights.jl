# Install necessary packages if not already installed
import Pkg
Pkg.add(["JuMP", "Gurobi", "StatsBase", "DataFrames", "Random", "CSV","LinearAlgebra"])
using JuMP, Gurobi, StatsBase, DataFrames, Random, CSV, LinearAlgebra
# --- Read in arguments ---
if length(ARGS) < 5
    error("Not enough arguments. Usage: julia acs_weights.jl model year split_type grouping_var rank_type iter")
end

model = ARGS[1]
year = ARGS[2]
split_type = ARGS[3]
grouping_var = ARGS[4]
rank_type = ARGS[5]
iter = parse(Int, ARGS[6])

println("Running: model=$model, year=$year, split_type=$split_type, grouping_var=$grouping_var, rank_type=$rank_type, iter=$iter")

# Construct file paths based on ACS format
file_basename = "$(model)_$(year)_$(split_type)_$(grouping_var)"
file_basename_with_iter = "$(file_basename)_$(iter)"

# Create output directory
output_dir = "data/acs_weights/$(file_basename)_$(rank_type)"
mkpath(output_dir)
output_path = joinpath(output_dir, "clustered_weights_seed_$(iter).csv")

if isfile(output_path)
    println("Output already exists at $output_path — skipping iteration $iter.")
    exit()
end

# --- Load data ---
global_pred_file = "predictions/test_global_$(file_basename_with_iter).csv"
backtest_info_file = "data/backtest_info_$(file_basename_with_iter).csv"
residual_pred_file = "predictions/test_local_$(file_basename_with_iter)_shifted_norms.csv"

global_pred_df = select(CSV.read(global_pred_file, DataFrame), Not(1))
backtest_cohort_df = select(CSV.read(backtest_info_file, DataFrame), Not(1))
residual_pred_df = select(CSV.read(residual_pred_file, DataFrame), Not(1))

# --- Setup ---
full_M = size(residual_pred_df, 2)
M=10
group_column = backtest_cohort_df[:, grouping_var]
group_counts = countmap(group_column)
sorted_loc_names = sort(collect(keys(group_counts)))
n_locations = length(sorted_loc_names)
results_matrix = zeros(n_locations, full_M)

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

# --- Run appropriate rank type ---
if rank_type in ["complex_opt_weights", "shifted_complex_opt"]
    for loc_idx in 1:n_locations
        current_loc = sorted_loc_names[loc_idx]
        println("Solving for location $current_loc")
        all_other_idxs = setdiff(1:n_locations, [loc_idx])
        random_sample_idxs = sort(all_other_idxs[randperm(length(all_other_idxs))[1:M-1]])
        selected_idxs = sort([loc_idx; random_sample_idxs])
        nA = group_counts[current_loc]

        idxs = backtest_cohort_df[:, grouping_var] .== current_loc

        y = backtest_cohort_df[idxs, 2] .-
            global_pred_df[idxs, current_loc]

        p = residual_pred_df[idxs, selected_idxs]

        if rank_type == "shifted_complex_opt"
            y .-= mean(y)
            # for j in 1:size(p, 2)
            #     p[:, j] .-= mean(p[:, j])
            # end
        end

        nj_subset = Dict(j => group_counts[sorted_loc_names[selected_idxs[j]]] for j in 1:M)
        local_idx = findfirst(x -> x == loc_idx, selected_idxs)

        z_values = solve_NLMIP(local_idx, M, nA, y, p, nj_subset)

        for j in 1:M
            global_idx = selected_idxs[j]
            results_matrix[loc_idx, global_idx] = z_values[j]
        end
    end

elseif rank_type in ["norms", "shifted_norms"]
    for i in 1:n_locations
        current_loc = sorted_loc_names[i]
        println("Solving for location $current_loc")
        idxs = backtest_cohort_df[:, grouping_var] .== current_loc

        y = backtest_cohort_df[idxs, 2] .-
            global_pred_df[idxs, current_loc]

        row_norms = zeros(n_locations)
        for j in 1:n_locations
            pred = residual_pred_df[idxs, j]

            if rank_type == "shifted_norms"
                y_shifted = y .- mean(y)
                pred_shifted = pred .- mean(pred)
                row_norms[j] = norm(y_shifted .- pred_shifted)
            else
                row_norms[j] = norm(y .- pred)
            end
        end
        row_norms[i] = 0
        println(row_norms)
        ranks = tiedrank(row_norms)
        println(ranks)
        results_matrix[i, :] = ranks
    end
elseif rank_type == "puc"
    for i in 1:n_locations
        current_loc = sorted_loc_names[i]
        println("Solving for location $current_loc (PUC)")
        idxs = backtest_cohort_df[:, grouping_var] .== current_loc

        scores = zeros(n_locations)
        p_tt_raw = residual_pred_df[idxs, i]
        p_tt = p_tt_raw .- mean(p_tt_raw)
        n_t = group_counts[current_loc]

        for j in 1:n_locations
            if j == i
                scores[j] = 0.0  # ensure current location is best
                continue
            end

            p_st_raw = residual_pred_df[idxs, j]
            p_st = p_st_raw .- mean(p_st_raw)

            n_s = group_counts[sorted_loc_names[j]]

            mse = mean((p_st .- p_tt).^2)
            avg_var = 0.5 * (var(p_st) + var(p_tt))
            penalty = max(mse / avg_var - 1 / n_s - 1 / n_t, 0)
            score = (1 / (n_s + n_t)) + penalty * (n_s^2) / (n_s + n_t)^2
            scores[j] = score
        end
        println(scores)
        ranks = tiedrank(scores)
        results_matrix[i, :] = ranks
    end
else
    error("Unknown rank_type: $rank_type. Must be one of: complex_opt_weights, shifted_complex_opt, norms, shifted_norms,puc.")
end

# --- Save Results ---
println("Results matrix for rank_type = $rank_type:")
results_matrix = DataFrame(results_matrix, Symbol.(sorted_loc_names))
CSV.write(output_path, results_matrix)

