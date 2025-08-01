rm(list=ls())
library(readr)
library(dplyr)

dataset <- "swiss"

architectures <- c("regression", "tree", "ranger", "bart")
methods <- c("global", "xlearn", "local", "grouped_xlearn", "jtt", "rwg")
archs = architectures
models = methods

ground_truth_df = read_csv(paste0("results/model_comparison/", dataset, "/", "global", "_", "bart", "_", dataset, "_shifted_predictions.csv"))

percentages <- c(10, 20, 30, 40, 50)
results_df <- data.frame()

for (arch in archs) {
  for (percent in percentages) {
    
    # First pass: collect top-k counts per group per model
    group_counts <- list()
    
    for (model in models) {
      predictions = read_csv(paste0("results/model_comparison/", dataset, "/", model, "_", arch, "_", dataset, "_shifted_all_predictions.csv"))
      
      top_indices <- lapply(predictions, function(col) {
        order(col, decreasing = TRUE)[1:as.integer((percent / 100) * length(col))]
      })
      
      for (colname in names(top_indices)) {
        top_rows <- ground_truth_df[top_indices[[colname]], ]
        matched_rows <- top_rows[top_rows$group == colname, ]
        
        if (!colname %in% names(group_counts)) {
          group_counts[[colname]] <- list()
        }
        group_counts[[colname]][[model]] <- nrow(matched_rows)
      }
    }
    
    # Keep only groups where all models have >=10 samples in top-k
    valid_groups <- names(group_counts)[sapply(group_counts, function(x) all(unlist(x) >= 10))]
    
    # Second pass: compute average outcome for valid groups only
    for (model in models) {
      predictions = read_csv(paste0("results/model_comparison/", dataset, "/", model, "_", arch, "_", dataset, "_shifted_all_predictions.csv"))
      
      top_indices <- lapply(predictions, function(col) {
        order(col, decreasing = TRUE)[1:as.integer((percent / 100) * length(col))]
      })
      
      total_value <- 0
      total_rows <- 0
      
      for (colname in names(top_indices)) {
        if (!(colname %in% valid_groups)) next
        
        top_rows <- ground_truth_df[top_indices[[colname]], ]
        matched_rows <- top_rows[top_rows$group == colname, ]
        total_value <- total_value + sum(matched_rows$ground_truth, na.rm = TRUE)
        total_rows <- total_rows + nrow(matched_rows)
      }
      
      result <- if (total_rows > 0) total_value / total_rows else NA
      
      results_df <- rbind(results_df, data.frame(
        model = model,
        arch = arch,
        top_percent = percent,
        avg_outcome = result,
        total_rows = total_rows
      ))
    }
  }
}

write_csv(results_df, paste0("results/model_comparison/", dataset, "/", "avg_outcome_top_matches.csv"))
