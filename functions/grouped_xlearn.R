grouped_xlearn <- function(model, training_cohort, backtest_cohort, predictors, var_outcome, weight_file, grouping_var, remove_shift,data_type) {
  library(pbapply)
  library(readr)
  library(parallel)
  library(ggplot2)
  library(ggforce)
  
  all.locs <- sort(union(unique(backtest_cohort[[grouping_var]]), unique(training_cohort[[grouping_var]])))
  print(all.locs)
  # Construct file paths
  file_base <- paste0(model, "_", data_type)
  file_base_with_group <- paste0(file_base, "_", grouping_var)
  if (remove_shift) {
    file_base_with_group <- paste0(file_base_with_group, "_shifted")
  }
  
  # Read weights file
  weights = read_csv(paste0('checkpoints/weights/',weight_file,".csv"))
  weights = weights[,all.locs]
  print(weights)
  if (length(all.locs) != ncol(weights)) {
    stop(paste0("Mismatch: length(all.locs) = ", length(all.locs), 
                " but ncol(weights) = ", ncol(weights)))
  }
  
  # Check for existing global predictions
  predictions_file <- paste0("predictions/global_", file_base, ".csv")
  predictions_test_file <- paste0("predictions/test_global_", file_base, ".csv")
  
  if (!file.exists(predictions_file) || !file.exists(predictions_test_file)) {
    # Train global model using ACS data
    g.out <- global_estimator(
      model=model,
      Lframe=as.data.frame(training_cohort), 
      predictors=predictors,
      outcome.var=var_outcome
    )
  }
  
  if (!file.exists(predictions_file)) {
    # Predict for training cohort using the global model
    if (model == "bart") {
      global_predictions <- g.out$Model_store$global$yhat.train.mean
    } else if (model == "ranger") {
      global_predictions <- g.out$Model_store$global$predictions
    } else {
      global_predictions <- predict_global(
        model=model,
        newdata=training_cohort,
        object=g.out,
        predictors=predictors,
        grouping_var=grouping_var,
        single_vector=TRUE
      )
    }
    write.csv(data.frame(x = global_predictions), predictions_file, row.names = FALSE)
  } else {
    global_predictions <- read.csv(predictions_file)$x
  }
  
  residuals <- numeric(length(global_predictions))
  
  # Calculate residuals
  for (k in 1:length(global_predictions)) {
    residuals[k] <- training_cohort[[var_outcome]][k] - 
      as.numeric(global_predictions[k])
  }
  
  # Handle shifts if needed
  if (remove_shift) {
    group_means <- tapply(residuals, training_cohort[[grouping_var]], mean)
    for (i in seq_along(residuals)) {
      group <- training_cohort[[grouping_var]][i]
      residuals[i] <- residuals[i] - group_means[group]
    }
  }
  
  training_cohort$residuals <- residuals
  training_cohort_with_residuals <- training_cohort
  
  # Process result files from the new ACS directory structure
  base_dir <- paste0('checkpoints/rmses/',weight_file)
  results <- data.frame()
  print(base_dir)
  # Loop through each group/location directory
  for (group in all.locs) {
    group_dir <- file.path(base_dir, paste0("group_", group))
    if (!dir.exists(group_dir)) next
    
    # Get all top_k directories for this group
    top_k_dirs <- list.files(group_dir, pattern = "^top_k_", full.names = TRUE)
    
    for (top_k_dir in top_k_dirs) {
      # Extract top_k value from directory name
      top_k <- as.numeric(gsub(".*top_k_([0-9]+)$", "\\1", top_k_dir))
      
      # Read all rmse files for this top_k
      rmse_files <- list.files(top_k_dir, pattern = "rmse_iter_.*\\.txt$", full.names = TRUE)
      rmse_values <- sapply(rmse_files, function(f) as.numeric(readLines(f)))
      
      # Calculate mean and standard error
      mean_rmse <- mean(rmse_values, na.rm = TRUE)
      se_rmse <- sd(rmse_values, na.rm = TRUE) / sqrt(length(rmse_values))
      
      # Add to results dataframe
      results <- rbind(results, data.frame(
        loc = group,
        top_k = top_k,
        mean_rmse = mean_rmse,
        se_rmse = se_rmse
      ))
    }
  }
  # Logic to determine top_k
  find_min_top_k <- function(df) {
    df %>%
      group_by(loc) %>%
      mutate(
        min_rmse = min(mean_rmse, na.rm = TRUE),
        threshold_rmse = min_rmse + se_rmse[which.min(mean_rmse)]
      ) %>%
      filter(mean_rmse <= threshold_rmse) %>%
      arrange(top_k) %>%  # Sort by top_k to get the smallest value that meets threshold
      slice(1) %>%
      select(loc, top_k, mean_rmse, se_rmse)
  }
  
  min_top_k <- find_min_top_k(results)
  ranks = min_top_k$top_k
  top_k_weights = weights
  top_k_weights = top_k_weights*0
  for (i in 1:length(ranks)){
    top_indices <- order(as.numeric(weights[i,]), decreasing = TRUE)[1:ranks[i]]
    top_k_weights[i,top_indices] = 1
  }
  # write.csv(top_k_weights, paste0('data/', file_base_with_group, '_top_k_weights.csv'), row.names = FALSE)
  # write.csv(results, paste0('data/', file_base_with_group, '_combined_results.csv'), row.names = FALSE)
  
  gl <- grouped_local_estimator(
    model=model,
    Lframe=as.data.frame(training_cohort_with_residuals), 
    predictors=predictors,
    outcome.var='residuals',
    location_weights=top_k_weights,
    grouping_var=grouping_var
  )
  
  if (!file.exists(predictions_test_file)) {
    M_global <- predict_global(
      model=model,
      newdata=backtest_cohort,
      object=g.out,
      predictors=predictors,
      grouping_var=grouping_var
    )
    colnames(M_global) <- all.locs
    write_csv(as.data.frame(M_global), predictions_test_file)
  } else {
    M_global <- read_csv(predictions_test_file)
    M_global <- M_global[,all.locs]
  }
  
  M_residual <- predict_local(
    model=model,
    newdata=backtest_cohort,
    object=gl,
    predictors=predictors,
    outcome.var='residuals',
    grouping_var=grouping_var
  )
  colnames(M_residual) <- all.locs
  # Combine global and residual predictions
  combined_predictions <- M_global + M_residual
  if (remove_shift) {
    for (loc in all.locs) {
      combined_predictions[, loc] <- combined_predictions[, loc] + group_means[loc]
    }
  }
  
  combined_predictions[combined_predictions > 1] = 1
  combined_predictions[combined_predictions < 0] = 0
  
  items_per_page <- 12  # Number of locations per page
  total_pages <- ceiling(length(unique(results$loc)) / items_per_page)
  
  # create directory if it doesn't exist
  if (!dir.exists(paste0("plots/", data_type))) {
    dir.create(paste0("plots/", data_type), recursive = TRUE)
  }
  pdf(paste0("plots/", data_type, '/', file_base_with_group, '.pdf'), width = 12, height = 10)
  
  # Loop through each page and generate plots
  for (page in 1:total_pages) {
    plot <- ggplot(results, aes(x = top_k, y = mean_rmse, color = loc)) +
      geom_point() +
      geom_line() +
      geom_errorbar(aes(ymin = mean_rmse - se_rmse, ymax = mean_rmse + se_rmse), width = 0.2) +
      facet_wrap_paginate(~ loc, scales = "free_y", ncol = 3, nrow = 4, page = page) +
      geom_vline(data = min_top_k, aes(xintercept = top_k, color = loc), linetype = "dashed") +
      labs(
        title = paste("RMSE Across Locations - Page", page, "of", total_pages),
        x = "Top K",
        y = "Mean RMSE"
      ) +
      theme_minimal() +
      theme(
        legend.position = "none"
      )
    
    print(plot)
  }
  
  dev.off()
  
  list(global_predictions = M_global, residual_predictions = M_residual, combined_predictions = combined_predictions)
} 