rm(list=ls())

# Set working directory
setwd("/n/home05/gjain/projects/refugee/GeomatchCode/")
dir <- paste0("/n/home05/gjain/projects/refugee/GeomatchCode/")

# Source required functions
source(paste0(dir, "functions/get_acs_data.R"))
source(paste0(dir, "functions/global_estimator_acs.R"))
source(paste0(dir, "functions/local_estimator_acs.R"))
source(paste0(dir, "functions/predict_global_acs.R"))
source(paste0(dir, "functions/predict_local_acs.R"))
source(paste0(dir, "functions/grouped_local_estimator_acs.R"))
source(paste0(dir, "functions/acs_model_xlearn.R"))
source(paste0(dir, "functions/grouped_xlearn_acs.R"))
source(paste0(dir, "functions/get_global_or_local_acs.R"))

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
prediction_type <- as.character(parsed_args[['prediction_type']])  # xlearn, grouped_xlearn, local, or global
grouping_var <- as.character(parsed_args[['grouping_var']])
rank_type <- as.character(parsed_args[['rank_type']])
outcome_var <- as.character(parsed_args[['outcome_var']])
model <- as.character(parsed_args[['model']])
year <- as.integer(parsed_args[['year']])
split_type <- as.character(parsed_args[['split_type']])

# Set up file paths and names
if (prediction_type == "grouped_xlearn") {
  base <- paste(model, rank_type, year, split_type, sep="_")
} else {
  base <- paste(model, year, split_type, sep="_")
}
data_filename <- 'data/education_dataset.csv'

# Get initial cohorts
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

training_cohort <- cohorts$training_cohort
backtest_cohort <- cohorts$backtest_cohort
predictors <- setdiff(colnames(training_cohort), c(outcome_var, 'Age','Speaks_Non_English','X'))

# Function to extract predictions for actual groups
extract_group_predictions <- function(predictions, actual_groups, grouping_var) {
  n <- nrow(predictions)
  extracted_preds <- numeric(n)
  for (i in 1:n) {
    extracted_preds[i] <- as.numeric(predictions[i, as.character(actual_groups[i])])
  }
  return(extracted_preds)
}

# Add forecast metric function
get_forecast_metric <- function(predictions, ground_truth, k = 10, 
                              binning_method = "width", metric = "calibration") {
  # Ensure inputs are numeric vectors
  predictions <- as.numeric(predictions)
  ground_truth <- as.numeric(ground_truth)
  N <- length(predictions)  # Total number of samples
  
  # Initialize bin indices
  bin_indices <- numeric(N)
  
  if (binning_method == "width") {
    # Equal width binning
    bin_edges <- seq(0, 1, length.out = k + 1)
    bin_indices <- cut(predictions, breaks = bin_edges, include.lowest = TRUE, labels = FALSE)
  } else if (binning_method == "size") {
    # Equal size binning (quantile-based)
    quantiles <- quantile(predictions, probs = seq(0, 1, length.out = k + 1), na.rm = TRUE)
    bin_indices <- cut(predictions, breaks = quantiles, include.lowest = TRUE, labels = FALSE)
  } else {
    stop("Invalid binning_method. Choose 'width' (equal-width bins) or 'size' (equal-size bins).")
  }
  
  # Overall mean of ground truth (used in resolution)
  o <- mean(ground_truth)
  
  # Initialize metric value
  metric_value <- 0
  
  for (i in 1:k) {
    # Get indices of samples in this bin
    bin_mask <- which(bin_indices == i)
    n_k <- length(bin_mask)  # Count of values in bin
    
    if (n_k > 0) {  # Avoid division by zero
      p_k <- mean(predictions[bin_mask])  # Average predicted value for bin
      o_k <- mean(ground_truth[bin_mask])  # Average observed value for bin
      
      if (metric == "calibration") {
        metric_value <- metric_value + n_k * (p_k - o_k)^2
      } else if (metric == "resolution") {
        metric_value <- metric_value + n_k * (o_k - o)^2
      } else {
        stop("Invalid metric. Choose 'calibration' or 'resolution'.")
      }
    }
  }
  
  # Final metric value
  return(metric_value / N)
}

# Helper function to calculate and save metrics
calculate_and_save_metrics <- function(predictions, ground_truth, groups, output_dir, base_filename) {
  # Save prediction vectors
  print(length(predictions))
  print(length(ground_truth))
  write.csv(data.frame(
    predictions = predictions,
    ground_truth = ground_truth,
    group = groups
  ), paste0(output_dir, "/", base_filename, "_predictions.csv"), row.names = FALSE)
  
  # Calculate group sizes
  group_sizes <- table(groups)
  small_groups <- names(group_sizes)[group_sizes <= quantile(group_sizes, 0.33)]
  small_mask <- groups %in% small_groups
  
  # Calculate metrics
  metrics <- list()
  
  # Overall MSE
  metrics$mse <- mean((ground_truth - predictions)^2)
  
  # Small group MSE
  metrics$small_mse <- mean((ground_truth[small_mask] - predictions[small_mask])^2)
  
  # Resolution (overall and small groups)
  metrics$resolution <- get_forecast_metric(predictions, ground_truth, 
                                        k = 10, binning_method = "width", metric = "resolution")
  metrics$small_resolution <- get_forecast_metric(predictions[small_mask], 
                                              ground_truth[small_mask], 
                                              k = 10, binning_method = "width", metric = "resolution")
  
  # Calibration (overall and small groups)
  metrics$calibration <- get_forecast_metric(predictions, ground_truth, 
                                         k = 10, binning_method = "width", metric = "calibration")
  metrics$small_calibration <- get_forecast_metric(predictions[small_mask], 
                                               ground_truth[small_mask], 
                                               k = 10, binning_method = "width", metric = "calibration")
  
  # Save metrics
  write.csv(data.frame(
    metric = names(metrics),
    value = unlist(metrics)
  ), paste0(output_dir, "/", base_filename, "_metrics.csv"), row.names = FALSE)
  
  return(metrics)
}

# Get predictions based on prediction_type
predictions <- NULL
if (prediction_type == "xlearn") {
  # Check if results already exist
  output_dir <- paste0("results/model_comparison/", grouping_var, "/", outcome_var)
  xlearn_filename <- paste("xlearn", base, sep="_")
  global_filename <- paste("global", base, sep="_")
  predictions_file <- paste0(output_dir, "/", xlearn_filename, "_predictions.csv")
  metrics_file <- paste0(output_dir, "/", xlearn_filename, "_metrics.csv")
  global_predictions_file <- paste0(output_dir, "/", global_filename, "_predictions.csv")
  global_metrics_file <- paste0(output_dir, "/", global_filename, "_metrics.csv")
  # Only run if any of the required files don't exist
  if (!all(file.exists(c(predictions_file, metrics_file, global_predictions_file, global_metrics_file)))) {
    result <- acs_model_xlearn(
      model = model,
      training_cohort = training_cohort,
      backtest_cohort = backtest_cohort,
      predictors = predictors,
      var_outcome = outcome_var,
      grouping_var = grouping_var
    )
    predictions <- result$combined_predictions
    
    # Calculate metrics for both global and combined predictions
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
    
    # Global predictions
    global_predictions <- result$global_predictions
    global_actual_predictions <- extract_group_predictions(global_predictions, backtest_cohort[[grouping_var]], grouping_var)
    calculate_and_save_metrics(
      predictions = global_actual_predictions,
      ground_truth = backtest_cohort[[outcome_var]],
      groups = backtest_cohort[[grouping_var]],
      output_dir = output_dir,
      base_filename = global_filename
    )
  } else {
    cat("Results already exist for xlearn. Skipping...\n")
  }
  
} else if (prediction_type == "grouped_xlearn") {
  # Check if results already exist
  output_dir <- paste0("results/model_comparison/", grouping_var, "/", outcome_var)
  grouped_filename <- paste("grouped_xlearn", base, sep="_")
  predictions_file <- paste0(output_dir, "/", grouped_filename, "_predictions.csv")
  metrics_file <- paste0(output_dir, "/", grouped_filename, "_metrics.csv")
  
  # Only run if any of the required files don't exist
  if (!all(file.exists(c(predictions_file, metrics_file)))) {
    result <- grouped_xlearn_acs(
      model = model,
      training_cohort = training_cohort,
      backtest_cohort = backtest_cohort,
      predictors = predictors,
      var_outcome = outcome_var,
      weight_file = paste0('acs_weights/regression', "_", year, "_", split_type, "_", grouping_var, "_", rank_type),
      year = year,
      split_type = split_type,
      grouping_var = grouping_var,
      rank_type = rank_type
    )
    predictions <- result$combined_predictions
  } else {
    cat("Results already exist for grouped_xlearn with rank_type", rank_type, ". Skipping...\n")
  }
  
} else if (prediction_type == "local") {
  # Check if results already exist
  output_dir <- paste0("results/model_comparison/", grouping_var, "/", outcome_var)
  local_filename <- paste("local", base, sep="_")
  predictions_file <- paste0(output_dir, "/", local_filename, "_predictions.csv")
  metrics_file <- paste0(output_dir, "/", local_filename, "_metrics.csv")
  
  # Only run if any of the required files don't exist
  if (!all(file.exists(c(predictions_file, metrics_file)))) {
    predictions <- get_global_or_local_acs(
      model = model,
      training_cohort = training_cohort,
      backtest_cohort = backtest_cohort,
      predictors = predictors,
      var_outcome = outcome_var,
      model_type = "local",
      grouping_var = grouping_var
    )
  } else {
    cat("Results already exist for local. Skipping...\n")
  }
} 

# Only calculate metrics if we have predictions
if (!is.null(predictions)) {
  # Extract predictions for actual groups
  actual_predictions <- extract_group_predictions(predictions, backtest_cohort[[grouping_var]], grouping_var)
  
  # Create output directory if it doesn't exist
  output_dir <- paste0("results/model_comparison/", grouping_var, "/", outcome_var)
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Calculate metrics using helper function
  out_filename <- switch(prediction_type,
                        "xlearn" = paste("xlearn", base, sep="_"),
                        "grouped_xlearn" = paste("grouped_xlearn", base, sep="_"),
                        "local" = paste("local", base, sep="_"))
  
  metrics_file <- paste0(output_dir, "/", out_filename, "_metrics.csv")
  
  # Only calculate metrics if the metrics file doesn't exist
  if (!file.exists(metrics_file)) {
    calculate_and_save_metrics(
      predictions = actual_predictions,
      ground_truth = backtest_cohort[[outcome_var]],
      groups = backtest_cohort[[grouping_var]],
      output_dir = output_dir,
      base_filename = out_filename
    )
  } else {
    cat("Metrics already exist for", prediction_type, ". Skipping calculation...\n")
  }
} else {
  # If predictions is NULL, check if we can load predictions from file
  output_dir <- paste0("results/model_comparison/", grouping_var, "/", outcome_var)
  out_filename <- switch(prediction_type,
                        "xlearn" = paste("xlearn", base, sep="_"),
                        "grouped_xlearn" = paste("grouped_xlearn", base, sep="_"),
                        "local" = paste("local", base, sep="_"))
  
  predictions_file <- paste0(output_dir, "/", out_filename, "_predictions.csv")
  metrics_file <- paste0(output_dir, "/", out_filename, "_metrics.csv")
  
  if (file.exists(predictions_file) && !file.exists(metrics_file)) {
    # Load predictions from file
    pred_data <- read.csv(predictions_file)
    calculate_and_save_metrics(
      predictions = pred_data$predictions,
      ground_truth = pred_data$ground_truth,
      groups = pred_data$group,
      output_dir = output_dir,
      base_filename = out_filename
    )
  } else if (!file.exists(predictions_file)) {
    cat("No predictions available for", prediction_type, ". Skipping metrics calculation...\n")
  } else {
    cat("Metrics already exist for", prediction_type, ". Skipping calculation...\n")
  }
} 