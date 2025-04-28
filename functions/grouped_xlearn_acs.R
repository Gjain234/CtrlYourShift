grouped_xlearn_acs <- function(model, training_cohort, backtest_cohort, predictors, var_outcome, weight_file, backtest_locs=NULL, verbose=FALSE, rank_type="", grouping_var="STATE") {
  library(pbapply)
  library(readr)
  library(parallel)
  library(ggplot2)
  library(ggforce)
  
  all.locs <- sort(union(unique(backtest_cohort[[grouping_var]]), unique(training_cohort[[grouping_var]])))
  weights = read.csv(paste0("data/",weight_file,".csv"))
  weights = weights[,all.locs]
  if (length(all.locs) != ncol(weights)) {
    stop(paste0("Mismatch: length(all.locs) = ", length(all.locs), 
                " but ncol(weights) = ", ncol(weights)))
  }
  
  # Train global model using ACS data
  g.out <- global_estimator_acs(
    model=model,
    Lframe=as.data.frame(training_cohort), 
    predictors=predictors,
    outcome.var=var_outcome
  )
  
  # Predict for training cohort using the global model
  pout <- predict_global_acs(
    model=model,
    newdata=training_cohort,
    object=g.out,
    states=all.locs,
    predictors=predictors,
    grouping_var=grouping_var
  )
  
  global_predictions <- pout$prob.test.mean
  residuals <- numeric(nrow(global_predictions))
  
  # Calculate residuals
  for (k in 1:nrow(global_predictions)) {
    residuals[k] <- training_cohort[[var_outcome]][k] - 
      as.numeric(global_predictions[k, as.character(training_cohort[[grouping_var]][k])])
  }
  
  # Handle shifts if needed
  remove_shift <- grepl("shift", rank_type)
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
  base_dir <- paste0("results/grouped/acs_weights/", weight_file)
  results <- data.frame()
  
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
    if (rank_type == "" | rank_type == "shifted_norms") {
      top_indices <- order(as.numeric(weights[i,]))[1:ranks[i]]
    } else {
      top_indices <- order(as.numeric(weights[i,]), decreasing = TRUE)[1:ranks[i]]
    }
    top_k_weights[i,top_indices] = 1
  }
  #   write.csv(top_k_weights, paste0('data/',weight_file,'_top_k_weights.csv'), row.names = FALSE)
  #   write.csv(results, paste0('data/',weight_file,'_combined_results.csv'), row.names = FALSE)
  
  if (is.null(backtest_locs)){
    backtest_locs = all.locs
  }
  
  gl <- grouped_local_estimator_acs(
    model=model,
    Lframe=as.data.frame(training_cohort_with_residuals), 
    predictors=predictors,
    outcome.var='residuals',
    location_weights=top_k_weights,
    states_to_use=NULL,
    single_state=FALSE,
    grouping_var=grouping_var
  )
  
  # Predict for backtest cohort using global and residual models
  pout_global <- predict_global_acs(
    model=model,
    newdata=backtest_cohort,
    object=g.out,
    states=all.locs,
    predictors=predictors,
    grouping_var=grouping_var,
    verbose=verbose
  )
  M_global <- pout_global$prob.test.mean
  
  pout_residual <- predict_local_acs(
    model=model,
    newdata=backtest_cohort,
    object=gl,
    states=all.locs,
    predictors=predictors,
    outcome.var='residuals',
    grouping_var=grouping_var,
    verbose=verbose
  )
  
  M_residual <- pout_residual$prob.test.mean
  
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
  
  # Open a PDF device
  pdf(paste0("plots/",weight_file,'.pdf'), width = 12, height = 10)
  
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
  
  list(global_predictions = M_global, residual_predictions = M_residual, combined_predictions = combined_predictions, global_model_param = pout_global$model_summary_list, residual_model_param = pout_residual$model_summary_list)
} 