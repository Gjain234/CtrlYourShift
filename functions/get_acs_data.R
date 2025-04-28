#' Load and split ACS data
#'
#' @param year The year to use for temporal splitting
#' @param split_type The type of split to use ('sampled' or 'temporal')
#' @param data_filename The path to the data file
#' @param var_outcome The outcome variable name
#' @param grouping_var The variable to use for grouping (default: "STATE")
#' @param seed Random seed for reproducibility (default: 123)
#' @param smallest_data_size Minimum number of observations per group (default: 30)
#' @param synthetic Whether to use synthetic data (default: FALSE)
#' @param assessment_predictions Optional assessment predictions for synthetic data
#' @param max_year Maximum year to include in the data (default: 2025)
#'
#' @return A list containing:
#'   - training_cohort: The training data
#'   - backtest_cohort: The test data
get_acs_data <- function(year, 
                        split_type = 'sampled',
                        data_filename = 'data/education_dataset.csv',
                        var_outcome = 'Education',
                        grouping_var = "STATE",
                        seed = 123,
                        smallest_data_size = 40,
                        max_year = 2025,
                        sample_size=1) {
  
  # Load required libraries
  require(dplyr)
  require(readr)
  require(tidyr)
  
  # Load the dataset
  dataset = read_csv(data_filename)
  dataset = dataset[,-1]  # Remove first column (assuming it's an index)
  
  # Convert outcome to binary if needed
  if (var_outcome == 'Education') {
    dataset$Education = as.numeric(dataset$Education >= 16)
  }
  
  # Filter data based on year
  dataset = dataset[dataset$YEAR == year,]
  
  # Filter based on age if needed
  if (var_outcome == 'Education') {
    dataset = dataset[dataset$Age > 18,]
  }
  
  # Sample the dataset if sample_size < 1
  if (sample_size < 1) {
    set.seed(seed)
    dataset = dataset[sample(1:nrow(dataset), size = floor(sample_size * nrow(dataset))), ]
  }
  # if grouping_var is Race_2, then for all rows where Race_2 is 1, subsample only 10% to keep in the dataset, keep the rest as is
  if (grouping_var == 'Race_2') {
    temp_dataset = dataset[dataset$Race_2 == 'R1', ]
    temp_dataset = temp_dataset[sample(1:nrow(temp_dataset), size = floor(0.1 * nrow(temp_dataset))), ]
    dataset = rbind(dataset[dataset$Race_2 != 'R1', ], temp_dataset)
  }
  
  # Define factor predictors
  factor_predictors <- c(
    "STATE",
    "Race_1",
    "Race_2",
    "Race_3",
    "Citizenship",
    "Mobility_Status",
    "Cognitive_Difficulty",
    "Language_Spoken_Home",
    "Speaks_Non_English"
  )
  
  # Convert each predictor to factor with levels set to all unique values in the dataset
  for (var in factor_predictors) {
    if (var %in% colnames(dataset)) {
      dataset[[var]] <- factor(dataset[[var]], levels = sort(unique(dataset[[var]])))
    }
  }
  # Get predictors
  predictors = setdiff(colnames(dataset), c(var_outcome, 'Age'))
  
  # Split the data
  if (split_type == 'sampled') {
    set.seed(seed)
    train_index = sample(1:nrow(dataset), 0.8 * nrow(dataset))
    training_cohort = dataset[train_index,]
    backtest_cohort = dataset[-train_index,]
  } else {
    # Temporal split
    training_cohort = dataset[dataset$YEAR < year,]
    backtest_cohort = dataset[dataset$YEAR == year,]
  }
  
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