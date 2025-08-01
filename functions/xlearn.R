xlearn <- function(model, training_cohort, backtest_cohort, predictors, var_outcome, grouping_var) {
  library(pbapply)
  library(readr)
  library(parallel)
  all.locs <- sort(union(unique(backtest_cohort[[grouping_var]]),unique(training_cohort[[grouping_var]])))
  
  # Construct file paths using the same pattern as get_splits_predictions.R
  file_base <- paste0(model, "_", data_type)
  
  # Check for existing global predictions
  predictions_file <- paste0("predictions/global_", file_base, ".csv")
  predictions_test_file <- paste0("predictions/test_global_", file_base, ".csv")
  
  if (!file.exists(predictions_file) || !file.exists(predictions_test_file)) {
    # Train global model
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
      probs <- apply(g.out$Model_store$global$yhat.train, 2, pnorm)
      global_predictions <- colMeans(probs)
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
    global_predictions <- read_csv(predictions_file)$x
  }
  print(head(global_predictions))
  residuals <- numeric(length(global_predictions))
  for (i in seq_len(length(global_predictions))) {
    residuals[i] <- training_cohort[[var_outcome]][i] - as.numeric(global_predictions[i])
  }
  training_cohort$residuals <- residuals
  training_cohort_with_residuals = training_cohort
  
  # Train residual model
  l.out <- local_estimator(
    model=model,
    Lframe = as.data.frame(training_cohort_with_residuals),
    predictors = predictors,
    outcome.var = 'residuals',
    grouping_var = grouping_var
  )
  
  # Get global predictions for backtest cohort
  if (!file.exists(predictions_test_file)) {
    M_global <- predict_global(
      model=model,
      newdata=backtest_cohort,
      object=g.out,
      predictors=predictors,
      grouping_var=grouping_var
    )
    colnames(M_global) <- all.locs
    write.csv(M_global, predictions_test_file)
  } else {
    M_global <- read_csv(predictions_test_file)
    M_global <- M_global[,all.locs]
  }
  
  M_residual <- predict_local(
    model=model,
    newdata=backtest_cohort,
    object=l.out,
    predictors=predictors,
    outcome.var = 'residuals',
    grouping_var = grouping_var
  )
  
  colnames(M_residual) <- all.locs
  # Combine global and residual predictions
  combined_predictions <- M_global + M_residual
  combined_predictions[combined_predictions > 1] = 1
  combined_predictions[combined_predictions < 0] = 0
  print(nrow(combined_predictions))
  print(nrow(backtest_cohort))
  print(nrow(M_global))
  list(global_predictions = M_global, residual_predictions = M_residual, combined_predictions = combined_predictions)
}
