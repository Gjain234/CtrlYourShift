#' Load and split health data
#'
#' @param data_filename The path to the data file
#' @param grouping_var The grouping variable name
#' @param seed Random seed for reproducibility (default: 123)
#' @param smallest_data_size Minimum number of observations per group (default: 40)
#'
#' @return A list containing:
#'   - training_cohort: The training data
#'   - backtest_cohort: The test data
get_data <- function(data_filename = 'data/health_dataset.csv',
                          grouping_var = 'group',
                          seed = 123,
                          smallest_data_size = 40) {
  
  # Load required libraries
  require(dplyr)
  require(readr)
  require(tidyr)
  set.seed(seed)

  # Load the dataset
  dataset = read_csv(data_filename)
  set.seed(seed)

  train_indices = sample(1:nrow(dataset), size = 0.8 * nrow(dataset))
  training_cohort = dataset[train_indices,]
  backtest_cohort = dataset[-train_indices,]
  
  # Filter out groups with too few observations
  group_counts = table(training_cohort[[grouping_var]])
  valid_groups = names(group_counts[group_counts > smallest_data_size])
  
  training_cohort = training_cohort[training_cohort[[grouping_var]] %in% valid_groups,]
  backtest_cohort = backtest_cohort[backtest_cohort[[grouping_var]] %in% valid_groups,]
  
  # Return results
  return(list(
    training_cohort = training_cohort,
    backtest_cohort = backtest_cohort
  ))
} 