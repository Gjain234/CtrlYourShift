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
source(paste0(dir, "functions/grouped_local_estimator.R"))

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
iter <- as.integer(parsed_args[['iter']])
year <- as.integer(parsed_args[['year']])
force_retrain <- as.logical(parsed_args[['force_retrain']])
grouping_var <- as.character(parsed_args[['grouping_var']])
outcome_var <- as.character(parsed_args[['outcome_var']])
rank_type <- as.character(parsed_args[['rank_type']])
remove_shift <- grepl("shift", rank_type)
split_type <- as.character(parsed_args[['split_type']])
model <- as.character(parsed_args[['model']])

# Set up file paths and names
data_filename <- 'data/education_dataset.csv'
file_basename <- paste0(model, "_", year, "_", split_type, "_", grouping_var)
file_basename_with_iter <- paste0(file_basename, "_", iter)

folder_name <- paste0("data/acs_weights/", file_basename, "_", rank_type)
output_file <- paste0(folder_name, "/clustered_weights_seed_", iter, ".csv")

# Check if output file exists
if (file.exists(output_file) && !force_retrain) {
  cat("Output file already exists:", output_file, "\nSkipping computation.\n")
  quit(save = "no")
}
sample_size = 1
if (grouping_var == 'STATE'){
  sample_size = 0.25
}
# Load and prepare data
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
predictors <- setdiff(colnames(temp_training_cohort), c(outcome_var, 'Age','Speaks_Non_English',"X"))

# Split data
set.seed(iter)
sample_train_index <- sample(1:nrow(temp_training_cohort), 0.8 * nrow(temp_training_cohort))
backtest_cohort <- temp_training_cohort[-sample_train_index,]
training_cohort <- temp_training_cohort[sample_train_index,]

# Save backtest info
backtest_info_file <- paste0("data/backtest_info_", file_basename_with_iter, ".csv")
if (!file.exists(backtest_info_file)) {
  write.csv(backtest_cohort[, c(grouping_var, outcome_var)], backtest_info_file)
}

# Train global model
if (!dir.exists("predictions")) {
  dir.create("predictions")
}

predictions_file <- paste0("predictions/global_", file_basename_with_iter, ".csv")

if (file.exists(predictions_file) && !force_retrain) {
  global_predictions <- as.matrix(read.csv(predictions_file)[, -1])
} else {
  g.out <- global_estimator_acs(
    model = model,
    Lframe = as.data.frame(training_cohort),
    predictors = predictors,
    outcome.var = outcome_var
  )
  
  pout <- predict_global_acs(
    model = model,
    newdata = training_cohort,
    object = g.out,
    states = all_locs,
    predictors = predictors
  )
  global_predictions <- pout$prob.test.mean
  write.csv(global_predictions, predictions_file)
}

# Calculate residuals
residuals <- numeric(nrow(global_predictions))
for (i in seq_len(nrow(global_predictions))) {
  residuals[i] <- training_cohort[[outcome_var]][i] - 
    as.numeric(global_predictions[i, as.character(training_cohort[[grouping_var]][i])])
}

# Handle residual shifting
if (remove_shift) {
  group_means <- tapply(residuals, training_cohort[[grouping_var]], mean)
  for (i in seq_len(nrow(global_predictions))) {
    group <- training_cohort[[grouping_var]][i]
    residuals[i] <- residuals[i] - group_means[group]
  }
}

# Train local model
training_cohort$residuals <- residuals
training_cohort_with_residuals <- training_cohort

h.out <- local_estimator_acs(
  model = model,
  Lframe = as.data.frame(training_cohort_with_residuals),
  predictors = predictors,
  outcome.var = 'residuals',
  grouping_var = grouping_var
)

# Make predictions
predictions_local_test_file <- paste0("predictions/test_local_", file_basename_with_iter,"_",rank_type,".csv")

if (!file.exists(predictions_local_test_file) || force_retrain) {
  pout <- predict_local_acs(
    model = model,
    newdata = backtest_cohort,
    object = h.out,
    states = all_locs,
    predictors = predictors,
    outcome.var = 'residuals'
  )
  residual_predictions <- pout$prob.test.mean
  write.csv(residual_predictions, predictions_local_test_file)
} else {
  residual_predictions <- as.matrix(read.csv(predictions_local_test_file)[, -1])
}

# Make test predictions
predictions_test_file <- paste0("predictions/test_global_", file_basename_with_iter, ".csv")

if (!file.exists(predictions_test_file) || force_retrain) {
  pout <- predict_global_acs(
    model = model,
    newdata = backtest_cohort,
    object = g.out,
    states = all_locs,
    predictors = predictors
  )
  global_predictions_test <- pout$prob.test.mean
  write.csv(global_predictions_test, predictions_test_file)
} else {
  global_predictions_test <- as.matrix(read.csv(predictions_test_file)[, -1])
}

# Calculate weights
predictions <- residual_predictions
aid_order <- backtest_cohort[[grouping_var]]
aid_counts <- table(aid_order)
employment <- backtest_cohort[[outcome_var]]
n <- nrow(predictions)
m <- ncol(predictions)
weights <- matrix(0, nrow = m, ncol = m)

for (k in seq_along(all_locs)) {
  aid <- all_locs[k]
  aid_matched_people <- predictions[aid_order == aid, , drop = FALSE]
  aid_matched_people_global <- as.numeric(global_predictions_test[aid_order == aid, aid, drop = FALSE])
  y <- employment[aid_order == aid]
  M <- as.matrix(aid_matched_people)
  n_aid <- length(y)
  residual_ground_truth <- y - aid_matched_people_global
  
  if (remove_shift) {
    residual_ground_truth <- residual_ground_truth - mean(residual_ground_truth)
  }
  
  norms <- sapply(seq_len(m), function(loc) sum((residual_ground_truth - M[, loc])^2))
  norms[k] <- 0
  ranks <- rank(norms)
  weights[k, ] <- ranks
}

# Save weights
weights_table <- as.data.frame(weights)
colnames(weights_table) <- all_locs
rownames(weights_table) <- all_locs

if (!dir.exists(folder_name)) {
  dir.create(folder_name, recursive = TRUE)
}

write.csv(weights_table, output_file, row.names = FALSE)
