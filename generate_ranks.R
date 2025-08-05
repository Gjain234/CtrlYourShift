rm(list=ls())
# Source required functions
source(paste0("functions/get_data.R"))
source(paste0("functions/global_estimator.R"))
source(paste0("functions/local_estimator.R"))
source(paste0("functions/predict_global.R"))
source(paste0("functions/predict_local.R"))
source(paste0("functions/grouped_local_estimator.R"))

# Load required libraries
require(dplyr)
require(tidyr)
require(tidyverse)
require(parallel)
require(readr)

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
parsed_args <- list()

for (arg in args) {
  key_value <- strsplit(sub('^--', '', arg), "=")[[1]]
  if (length(key_value) == 2) {
    key <- key_value[1]
    value <- gsub("''", "'", key_value[2])
    parsed_args[[key]] <- value
  }
}

# Extract parameters
iter <- as.numeric(parsed_args[['iter']])
model <- as.character(parsed_args[['model']])
force_retrain <- as.logical(parsed_args[['force_retrain']])
outcome_var <- as.character(parsed_args[['outcome_var']])
grouping_var <- as.character(parsed_args[['grouping_var']])
top_k <- as.integer(parsed_args[['top_k']])
remove_shift <- as.logical(parsed_args[['remove_shift']])
data_type <- as.character(parsed_args[['data_type']])

# Set up file paths and names
data_filename <- paste0('data/', data_type, '_dataset.csv')
file_base <- paste0(model, "_", data_type)
file_base_with_iter <- paste0(file_base, "_", iter)
file_base_with_group <- paste0(file_base, "_", grouping_var)

cohorts <- get_data(
  data_filename = data_filename,
  grouping_var = grouping_var
)
predictors <- setdiff(colnames(cohorts$training_cohort), c(outcome_var, 'X'))
temp_training_cohort <- cohorts$training_cohort
temp_backtest_cohort <- cohorts$backtest_cohort
all_locs <- sort(union(unique(temp_backtest_cohort[[grouping_var]]), 
                      unique(temp_training_cohort[[grouping_var]])))

# Split data
set.seed(iter)
sample_train_index <- sample(1:nrow(temp_training_cohort), 0.8 * nrow(temp_training_cohort))
backtest_cohort <- temp_training_cohort[-sample_train_index,]
training_cohort <- temp_training_cohort[sample_train_index,]

# Load weights
weights_file <- paste0("checkpoints/weights/", file_base_with_group, if(remove_shift) "_shifted" else "", ".csv")
print(weights_file)
if (!file.exists(weights_file) || force_retrain) {
  data_dir <- paste0("checkpoints/weights/", file_base_with_group, if(remove_shift) "_shifted" else "")
  csv_files <- list.files(data_dir, pattern = "*\\.csv", full.names = TRUE)
  print(csv_files)
  print(all_locs)
  print(colnames(read_csv(csv_files[1])))
  first_df <- read_csv(csv_files[1])[,all_locs]
  sum_matrix <- matrix(0, nrow = nrow(first_df), ncol = ncol(first_df))
  
  for (file in csv_files) {
    df <- read_csv(file)[,all_locs]
    sum_matrix <- sum_matrix + df
  }
  
  weights <- sum_matrix / length(csv_files)
  rownames(weights) <- rownames(first_df)
  colnames(weights) <- colnames(first_df)
  write.csv(weights, weights_file)
} else {
  weights <- read_csv(weights_file)
}

weights <- weights[, all_locs]
top_k_weights <- as.data.frame(t(apply(weights, 1, function(row) {
  top_indices <- order(row, decreasing = TRUE)[1:top_k]
  as.numeric(seq_along(row) %in% top_indices)
})))

rownames(top_k_weights) <- rownames(weights)
colnames(top_k_weights) <- colnames(weights)

# Train global model if needed
predictions_file <- paste0("predictions/global_", file_base_with_iter, ".csv")
predictions_test_file <- paste0("predictions/test_global_", file_base_with_iter, ".csv")

if (!file.exists(predictions_file) || !file.exists(predictions_test_file)) {
  g.out <- global_estimator(
    model = model,
    Lframe = as.data.frame(training_cohort), 
    predictors = predictors,
    outcome.var = outcome_var
  )
}

if (!file.exists(predictions_file)) {
  global_predictions <- predict_global(
    model = model,
    newdata = training_cohort,
    object = g.out,
    states = all_locs,
    predictors = predictors,
    grouping_var = grouping_var,
    single_vector=TRUE
  )
  write.csv(global_predictions, predictions_file,row.names=FALSE)
} else {
  global_predictions <- read.csv(predictions_file)$x
}

# Calculate residuals
residuals <- numeric(length(global_predictions))
for (k in 1:length(global_predictions)) {
  residuals[k] <- training_cohort[[outcome_var]][k] - 
    as.numeric(global_predictions[k])
}

if (remove_shift) {
  group_means <- tapply(residuals, training_cohort[[grouping_var]], mean)
  for (i in seq_along(residuals)) {
    group <- training_cohort[[grouping_var]][i]
    residuals[i] <- residuals[i] - group_means[group]
  }
}

training_cohort$residuals <- residuals
training_cohort_with_residuals <- training_cohort

# Train local model once for all locations
gl <- grouped_local_estimator(
  model = model,
  Lframe = as.data.frame(training_cohort_with_residuals),
  predictors = predictors,
  outcome.var = 'residuals',
  location_weights = top_k_weights,
  grouping_var = grouping_var
)

# Make local predictions for all locations
residual_predictions <- predict_local(
  model = model,
  newdata = backtest_cohort,
  object = gl,
  predictors = predictors,
  outcome.var = 'residuals',
  grouping_var = grouping_var
)

if (!file.exists(predictions_test_file)) {
  global_predictions_test <- predict_global(
    model = model,
    newdata = backtest_cohort,
    object = g.out,
    predictors = predictors,
    grouping_var = grouping_var,
    single_vector=TRUE
  )
  write.csv(global_predictions_test, predictions_test_file,row.names=FALSE)
} else {
  global_predictions_test <- read.csv(predictions_test_file)$x
}
# Combine predictions for all locations
combined_predictions <- residual_predictions
for (loc in all_locs) {
  combined_predictions[, loc] <- combined_predictions[, loc] + global_predictions_test
}
if (remove_shift) {
  for (loc in all_locs) {
    combined_predictions[, loc] <- combined_predictions[, loc] + group_means[loc]
  }
}

# Clip predictions
combined_predictions[combined_predictions > 1] <- 1
combined_predictions[combined_predictions < 0] <- 0

# Create output directory structure
output_dir <- paste0("checkpoints/rmses/", file_base_with_group, if(remove_shift) "_shifted" else "")
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Compute and save location-specific RMSEs
for (loc in all_locs) {
  loc_indices <- backtest_cohort[[grouping_var]] == loc
  ground_truth <- backtest_cohort[[outcome_var]][loc_indices]
  predictions <- as.numeric(combined_predictions[loc_indices, loc])
  mse <- mean((ground_truth - predictions)^2)
  print(mse)
  mse_file <- paste0(output_dir, "/group_", loc,"/top_k_",top_k, "/rmse_iter_", iter, ".txt")
  dir.create(dirname(mse_file), recursive = TRUE, showWarnings = FALSE)
  writeLines(as.character(mse), mse_file)
}