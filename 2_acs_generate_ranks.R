rm(list=ls())

# Set working directory as needed
setwd("GeomatchCode/")
dir <- paste0("GeomatchCode/")

# Source required functions
source(paste0(dir, "functions/get_acs_data.R"))
source(paste0(dir, "functions/global_estimator_acs.R"))
source(paste0(dir, "functions/local_estimator_acs.R"))
source(paste0(dir, "functions/predict_global_acs.R"))
source(paste0(dir, "functions/predict_local_acs.R"))
source(paste0(dir, "functions/grouped_local_estimator_acs.R"))

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
year <- as.integer(parsed_args[['year']])
model <- as.character(parsed_args[['model']])
split_type <- as.character(parsed_args[['split_type']])
force_retrain <- as.logical(parsed_args[['force_retrain']])
rank_type <- as.character(parsed_args[['rank_type']])
outcome_var <- as.character(parsed_args[['outcome_var']])
grouping_var <- as.character(parsed_args[['grouping_var']])
top_k <- as.integer(parsed_args[['top_k']])
remove_shift <- grepl("shift", rank_type)
# Set up file paths and names
data_filename <- 'data/education_dataset.csv'
global_predictions_file <- paste0(model, "_", year, "_", split_type, "_", grouping_var, "_", iter)
file_basename <- paste0(model, "_", year, "_", split_type, "_", grouping_var, "_", rank_type)

# Load and prepare data
sample_size = 1
if (grouping_var == 'STATE'){
  sample_size = 0.25
}
cohorts <- get_acs_data(
  year = year,
  split_type = split_type,
  data_filename = data_filename,
  grouping_var = grouping_var,
  var_outcome = outcome_var,
  smallest_data_size = 40,
  sample_size = sample_size
)

temp_training_cohort <- cohorts$training_cohort
temp_backtest_cohort <- cohorts$backtest_cohort
all_locs <- sort(union(unique(temp_backtest_cohort[[grouping_var]]), 
                      unique(temp_training_cohort[[grouping_var]])))
predictors <- setdiff(colnames(temp_training_cohort), c(outcome_var, 'Age','Speaks_Non_English','X'))

# Split data
set.seed(iter)
sample_train_index <- sample(1:nrow(temp_training_cohort), 0.8 * nrow(temp_training_cohort))
backtest_cohort <- temp_training_cohort[-sample_train_index,]
training_cohort <- temp_training_cohort[sample_train_index,]

# Load weights
weights_file <- paste0("data/acs_weights/", file_basename, ".csv")
if (!file.exists(weights_file) || force_retrain) {
  data_dir <- paste0("data/acs_weights/", file_basename)
  csv_files <- list.files(data_dir, pattern = "*\\.csv", full.names = TRUE)
  first_df <- read.csv(csv_files[1])[,all_locs]
  sum_matrix <- matrix(0, nrow = nrow(first_df), ncol = ncol(first_df))
  
  for (file in csv_files) {
    df <- read.csv(file)[,all_locs]
    sum_matrix <- sum_matrix + df
  }
  
  weights <- sum_matrix / length(csv_files)
  rownames(weights) <- rownames(first_df)
  colnames(weights) <- colnames(first_df)
  write.csv(weights, weights_file)
} else {
  weights <- read.csv(weights_file, row.names = 1)
}

weights <- weights[, all_locs]
top_k_weights <- as.data.frame(t(apply(weights, 1, function(row) {
  top_indices <- if (grepl("norm", rank_type)) {
    order(row)[1:top_k]
  } else {
    order(row, decreasing = TRUE)[1:top_k]
  }
  as.numeric(seq_along(row) %in% top_indices)
})))

rownames(top_k_weights) <- rownames(weights)
colnames(top_k_weights) <- colnames(weights)

# Train global model if needed
predictions_file <- paste0("predictions/global_", global_predictions_file, ".csv")
predictions_test_file <- paste0("predictions/test_global_", global_predictions_file, ".csv")

if (!file.exists(predictions_file) || !file.exists(predictions_test_file)) {
  g.out <- global_estimator_acs(
    model = model,
    Lframe = as.data.frame(training_cohort), 
    predictors = predictors,
    outcome.var = outcome_var
  )
}

if (!file.exists(predictions_file)) {
  pout <- predict_global_acs(
    model = model,
    newdata = training_cohort,
    object = g.out,
    states = all_locs,
    predictors = predictors,
    grouping_var = grouping_var
  )
  global_predictions <- pout$prob.test.mean
  write.csv(global_predictions, predictions_file)
} else {
  global_predictions <- read.csv(predictions_file)
}

# Calculate residuals
global_predictions = global_predictions[, all_locs]
residuals <- numeric(nrow(global_predictions))
for (k in 1:nrow(global_predictions)) {
  residuals[k] <- training_cohort[[outcome_var]][k] - 
    as.numeric(global_predictions[k, as.character(training_cohort[[grouping_var]][k])])
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
gl <- grouped_local_estimator_acs(
  model = model,
  Lframe = as.data.frame(training_cohort_with_residuals),
  predictors = setdiff(predictors, c(grouping_var)),
  outcome.var = 'residuals',
  location_weights = top_k_weights,
  grouping_var = grouping_var
)

# Make local predictions for all locations
pout <- predict_local_acs(
  model = model,
  newdata = backtest_cohort,
  object = gl,
  states = all_locs,
  predictors = setdiff(predictors, c(grouping_var)),
  outcome.var = 'residuals',
  grouping_var = grouping_var
)
residual_predictions <- pout$prob.test.mean

if (!file.exists(predictions_test_file)) {
  pout_global <- predict_global_acs(
    model = model,
    newdata = backtest_cohort,
    object = g.out,
    states = all_locs,
    predictors = predictors,
    grouping_var = grouping_var
  )
  global_predictions_test <- pout_global$prob.test.mean
  write.csv(global_predictions_test, predictions_test_file)
} else {
  global_predictions_test <- read.csv(predictions_test_file)
}
global_predictions_test=global_predictions_test[, all_locs]
# Combine predictions for all locations
combined_predictions <- global_predictions_test + residual_predictions
if (remove_shift) {
  for (loc in all_locs) {
    combined_predictions[, loc] <- combined_predictions[, loc] + group_means[loc]
  }
}

# Clip predictions
combined_predictions[combined_predictions > 1] <- 1
combined_predictions[combined_predictions < 0] <- 0

# Create output directory structure
output_dir <- paste0("results/grouped/acs_weights/", file_basename)
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
