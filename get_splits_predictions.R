rm(list=ls())

setwd("/Users/gaurijain/Desktop/GeoMatchCode/cleaned_pipeline")
dir <- paste0("/Users/gaurijain/Desktop/GeoMatchCode/cleaned_pipeline/")

# Source required functions
source(paste0(dir, "functions/global_estimator.R"))
source(paste0(dir, "functions/local_estimator.R"))
source(paste0(dir, "functions/predict_global.R"))
source(paste0(dir, "functions/predict_local.R"))
source(paste0(dir, "functions/grouped_local_estimator.R"))
source(paste0(dir, "functions/get_data.R"))

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
force_retrain <- as.logical(parsed_args[['force_retrain']])
grouping_var <- as.character(parsed_args[['grouping_var']])
outcome_var <- as.character(parsed_args[['outcome_var']])
remove_shift <- as.logical(parsed_args[['remove_shift']])
model <- as.character(parsed_args[['model']])
data_type <- as.character(parsed_args[['data_type']])
data_filename <- paste0('data/', data_type, '_dataset.csv')
print(data_filename)

# Set up file paths and names
file_base <- paste0(model, "_", data_type)
file_base_with_iter <- paste0(file_base, "_", iter)
file_base_with_group <- paste0(file_base, "_", grouping_var)

# Check if output files already exist
predictions_file <- paste0("predictions/global_", file_base_with_iter, ".csv")
predictions_test_file <- paste0("predictions/test_global_", file_base_with_iter, ".csv")
predictions_local_test_file <- paste0("predictions/test_residual_", file_base_with_group, if(remove_shift) "_shifted" else "", "_", iter, ".csv")
backtest_info_file <- paste0("data/backtest_info_", file_base_with_group, "_", iter, ".csv")

if (!force_retrain && file.exists(predictions_file) && file.exists(predictions_test_file) && file.exists(predictions_local_test_file) && file.exists(backtest_info_file)) {
  cat("All output files already exist for iteration", iter, "- skipping computation.\n")
  cat("Files found:\n")
  cat("  -", predictions_file, "\n")
  cat("  -", predictions_test_file, "\n")
  cat("  -", predictions_local_test_file, "\n")
  cat("  -", backtest_info_file, "\n")
  quit(save = "no")
}

cohorts <- get_data(
  data_filename = data_filename,
  grouping_var = grouping_var)

temp_training_cohort <- cohorts$training_cohort
temp_backtest_cohort <- cohorts$backtest_cohort
all_groups <- sort(union(unique(temp_backtest_cohort[[grouping_var]]), 
                      unique(temp_training_cohort[[grouping_var]])))
predictors <- setdiff(colnames(temp_training_cohort), c(outcome_var,'X'))

set.seed(iter)
sample_train_index <- sample(1:nrow(temp_training_cohort), 0.8 * nrow(temp_training_cohort))
backtest_cohort <- temp_training_cohort[-sample_train_index,]
training_cohort <- temp_training_cohort[sample_train_index,]

# Save backtest info
if (!file.exists(backtest_info_file)) {
  write.csv(backtest_cohort[, c(grouping_var, outcome_var)], backtest_info_file,row.names=FALSE)
}

# Train global model
if (!dir.exists("predictions")) {
  dir.create("predictions")
}

if (file.exists(predictions_file) && file.exists(predictions_test_file) && !force_retrain) {
  global_predictions <- read_csv(predictions_file)$x
  global_predictions_test <- read_csv(predictions_test_file)$x
} else {
  g.out <- global_estimator(
    model = model,
    Lframe = as.data.frame(training_cohort),
    predictors = predictors,
    outcome.var = outcome_var
  )
  global_predictions <- predict_global(
    model = model,
    newdata = training_cohort,
    object = g.out,
    predictors = predictors,
    grouping_var = grouping_var,
    single_vector = TRUE
  )
  write.csv(global_predictions, predictions_file,row.names=FALSE)
  
  global_predictions_test <- predict_global(
    model = model,
    newdata = backtest_cohort,
    object = g.out,
    predictors = predictors,
    grouping_var = grouping_var,
    single_vector=TRUE
  )
  write.csv(global_predictions_test, predictions_test_file,row.names=FALSE)
}

# Make predictions
if (!file.exists(predictions_local_test_file) || force_retrain) {
  # Calculate residuals
  residuals <- numeric(length(global_predictions))
  for (i in seq_len(length(global_predictions))) {
    residuals[i] <- training_cohort[[outcome_var]][i] - as.numeric(global_predictions[i])
  }
  print(length(residuals))
  print(length(training_cohort[[grouping_var]]))
  # Handle residual shifting
  if (remove_shift) {
    group_means <- tapply(residuals, training_cohort[[grouping_var]], mean)
    for (i in seq_len(length(global_predictions))) {
      group <- training_cohort[[grouping_var]][i]
      residuals[i] <- residuals[i] - group_means[group]
    }
  }

  # Train local model
  training_cohort$residuals <- residuals
  training_cohort_with_residuals <- training_cohort
  h.out <- local_estimator(
    model = model,
    Lframe = as.data.frame(training_cohort_with_residuals),
    predictors = predictors,
    outcome.var = 'residuals',
    grouping_var = grouping_var
  )
  residual_predictions <- predict_local(
    model = model,
    newdata = backtest_cohort,
    object = h.out,
    predictors = predictors,
    outcome.var = 'residuals',
    grouping_var = grouping_var
  )
  write.csv(residual_predictions, predictions_local_test_file,row.names=FALSE)
}