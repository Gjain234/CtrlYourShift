require(readr)
require(dplyr)
require(tidyr)
require(tidyverse)
source(paste0("functions/get_data.R"))
source(paste0("functions/rwg_estimator.R"))
source(paste0("functions/predict_rwg.R"))
source(paste0("functions/jtt_estimator.R"))

data_type = "synthetic"
data_filename = paste0('data/', data_type, '_dataset.csv')
grouping_var = 'group'
# change this to outcome variable!
outcome_var = 'outcome'
cohorts <- get_data(
  data_filename = data_filename,
  grouping_var = grouping_var)

training_cohort <- cohorts$training_cohort
backtest_cohort <- cohorts$backtest_cohort
predictors <- setdiff(colnames(training_cohort), c(outcome_var))

n <- nrow(training_cohort)
group_counts <- table(training_cohort[[grouping_var]])
num_groups <- length(group_counts)
target_weight_per_group <- n / num_groups
weights <- as.numeric(target_weight_per_group / group_counts[training_cohort[[grouping_var]]])

# define a function call get_single_vector that takes a predictions dataframe and indexes the correct group value per row into a single vector
get_single_vector <- function(predictions_df) {
  single_vector <- numeric(nrow(predictions_df))
  for (i in 1:length(single_vector)){
    single_vector[i] = as.numeric(predictions_df[i, backtest_cohort[[grouping_var]][i]])
  }
  return(single_vector)
}

models <- c('regression', 'tree', 'ranger', 'bart')
results <- data.frame()

small_groups <- names(group_counts)[group_counts <= quantile(group_counts, 0.33)]

for (model in models) {
  # ---------- RWG ----------
  g.out <- rwg_estimator(
    model = model,
    Lframe = as.data.frame(training_cohort),
    predictors = predictors,
    outcome.var = outcome_var,
    weights = weights
  )
  
  all_groups <- sort(unique(training_cohort[[grouping_var]]))
  global_predictions <- predict_rwg(
    model = model,
    newdata = backtest_cohort,
    object = g.out,
    predictors = predictors,
    grouping_var = grouping_var,
    single_vector = FALSE,
    states = all_groups
  )
  
  write_csv(as.data.frame(global_predictions$prob.test.mean),
            paste0("results/model_comparison/",data_type,"/rwg_", model, "_", data_type,"_shifted_all_predictions.csv"))
  
  global_predictions_single_vector <- get_single_vector(global_predictions$prob.test.mean)
  mse_global <- mean((backtest_cohort[[outcome_var]] - global_predictions_single_vector)^2, na.rm = TRUE)
  
  small_group_rows <- backtest_cohort[[grouping_var]] %in% small_groups
  mse_global_small <- mean(
    (backtest_cohort[small_group_rows, outcome_var][[outcome_var]] - 
       global_predictions_single_vector[small_group_rows])^2,
    na.rm = TRUE
  )
  
  results <- bind_rows(results, data.frame(
    model = model,
    method = "rwg",
    mse_all = mse_global,
    mse_small = mse_global_small
  ))
  
  # ---------- JTT ----------
  g.out <- jtt_estimator(
    model = model,
    Lframe = as.data.frame(training_cohort),
    predictors = predictors,
    outcome.var = outcome_var
  )
  
  global_predictions <- predict_rwg(
    model = model,
    newdata = backtest_cohort,
    object = g.out,
    predictors = predictors,
    grouping_var = grouping_var,
    single_vector = FALSE,
    states = all_groups
  )
  
  write_csv(as.data.frame(global_predictions$prob.test.mean),
            paste0("results/model_comparison/",data_type,"/jtt_", model, "_", data_type,"_shifted_all_predictions.csv"))
  
  global_predictions_single_vector <- get_single_vector(global_predictions$prob.test.mean)
  mse_global <- mean((backtest_cohort[[outcome_var]] - global_predictions_single_vector)^2, na.rm = TRUE)
  
  small_group_rows <- backtest_cohort[[grouping_var]] %in% small_groups
  mse_global_small <- mean(
    (backtest_cohort[small_group_rows, outcome_var][[outcome_var]] - 
       global_predictions_single_vector[small_group_rows])^2,
    na.rm = TRUE
  )
  
  results <- bind_rows(results, data.frame(
    model = model,
    method = "jtt",
    mse_all = mse_global,
    mse_small = mse_global_small
  ))
}

# Optionally save results
write_csv(results, paste0("results/model_comparison/",data_type,"/baseline_mse_summary.csv"))
