#!/bin/bash
start_time=$(date +%s)

# ----- DETECT NUMBER OF CORES -----
if command -v nproc >/dev/null; then
  N_CORES=$(nproc)
elif [[ "$OSTYPE" == "darwin"* ]]; then
  N_CORES=$(sysctl -n hw.ncpu)
else
  echo "Cannot determine number of cores. Please set N_CORES manually."; exit 1
fi

# ----- CONFIGURATION -----
num_iter=250
max_top_k=10

export model="regression"
# THIS WILL WORK FOR A DATASET IN "data/jtpa_dataset.csv" that has outcome "employed" and group "site"
# Change your data name to "data/{data_type}_dataset.csv"
export data_type="synthetic"
export grouping_var="group"
export outcome_var="outcome"
export remove_shift=TRUE

mkdir -p logs

# # ========== STAGE 1 ==========
# logs in logs/weights_0.out, logs/weights_1.out, etc.
echo "Running: Generate Weights (R, 0-$((num_iter-1)))"
generate_weights() {
    iter=$1
    echo "Calling Rscript with: --iter=${iter} --model=${model} --data_type=${data_type} ..."
    Rscript get_splits_predictions.R \
      --iter=${iter} \
      --force_retrain=FALSE \
      --grouping_var=${grouping_var} \
      --outcome_var=${outcome_var} \
      --remove_shift=${remove_shift} \
      --model=${model} \
      --data_type=${data_type} \
      > logs/weights_${iter}.out 2>&1
}
export -f generate_weights
seq 0 $((num_iter-1)) | parallel -j $N_CORES generate_weights

# ========== STAGE 2 ==========
# logs in logs/julia_weights_0.out, logs/julia_weights_1.out, etc.
echo "Running: Julia Weight Script (0-$((num_iter-1)))"
run_julia_weights() {
    iter=$1
    julia generate_weights.jl "$model" "$data_type" "$grouping_var" "true" "$iter" > logs/julia_weights_${iter}.out 2>&1
}
export -f run_julia_weights
seq 0 $((num_iter-1)) | parallel -j $N_CORES run_julia_weights

# ========== STAGE 3 ==========
# logs in logs/ranks_0_1.out, logs/ranks_0_2.out, etc.
echo "Running: Generate Ranks (0-$((num_iter*max_top_k-1)))"
generate_ranks() {
    iter=$1
    top_k=$2
    echo "Calling Rscript with: --iter=${iter} --top_k=${top_k} --model=${model} --data_type=${data_type} ..."
    Rscript generate_ranks.R \
      --iter=${iter} \
      --model=${model} \
      --force_retrain=FALSE \
      --outcome_var=${outcome_var} \
      --grouping_var=${grouping_var} \
      --top_k=${top_k} \
      --remove_shift=${remove_shift} \
      --data_type=${data_type} \
      --shift=${remove_shift} \
      > logs/ranks_${iter}_${top_k}.out 2>&1
}

export -f generate_ranks
parallel -j $N_CORES generate_ranks ::: $(seq 0 $((num_iter - 1))) ::: $(seq 1 $max_top_k)

# ========== STAGE 4 ==========
# logs in logs/final_xlearn_bart.out, logs/final_grouped_xlearn_bart.out, etc.
echo "Running: Final Model Comparison (1-12)"
compare_final_models() {
    task_id=$1
    task_idx=$((task_id - 1))

    models=("bart" "regression" "tree" "ranger")
    prediction_types=("xlearn" "grouped_xlearn" "local")

    model_index=$((task_idx / 3))
    pred_type_index=$((task_idx % 3))

    model=${models[$model_index]}
    prediction_type=${prediction_types[$pred_type_index]}

    echo "Running model=$model, prediction_type=$prediction_type"

    Rscript compare_models.R \
      --prediction_type=${prediction_type} \
      --grouping_var=${grouping_var} \
      --outcome_var=${outcome_var} \
      --model=${model} \
      --data_type=${data_type} \
      --remove_shift=${remove_shift} \
      --backtest_all_data=FALSE \
      --shift=${remove_shift} \
      > logs/final_${prediction_type}_${model}.out 2>&1
}
export -f compare_final_models
seq 1 12 | parallel -j $N_CORES compare_final_models

# ========== STAGE 5 ==========
# logs in logs/model_comparisons.out
echo "Running: Plotting Model Comparisons"
Rscript plot_model_comparisons.R "$grouping_var" "$outcome_var" "$data_type" > logs/model_comparisons.out 2>&1

# ========== STAGE 6a ==========
# logs in logs/assign_xlearn_regression.out, logs/assign_local_regression.out, etc.
echo "Running: Parallel Assignment Computation (16 jobs)"
generate_assignment() {
    method=$1
    model=$2
    julia get_assignment.jl "$data_type" "$grouping_var" "$outcome_var" "${method}_${model}" > logs/assign_${method}_${model}.out 2>&1
}
export -f generate_assignment
parallel -j $N_CORES generate_assignment ::: xlearn local grouped_xlearn global ::: regression tree ranger bart

# ========== STAGE 6b ==========
# logs in logs/reward_computation.out
echo "Running: Reward Computation and Plotting"
julia get_reward.jl "$data_type" "$grouping_var" "$outcome_var" > logs/reward_computation.out 2>&1

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
hours=$((elapsed / 3600))
minutes=$(( (elapsed % 3600) / 60 ))
seconds=$((elapsed % 60))
printf "Total runtime: %02d:%02d:%02d (hh:mm:ss)\n" $hours $minutes $seconds
