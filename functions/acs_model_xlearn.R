acs_model_xlearn <- function(model, training_cohort, backtest_cohort, predictors, var_outcome, grouping_var = "STATE", verbose=FALSE) {
  library(pbapply)
  library(readr)
  library(parallel)
  all.locs <- sort(union(unique(backtest_cohort[[grouping_var]]),unique(training_cohort[[grouping_var]])))
  # Train global model
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
    predictors=predictors
  )
  
  predictions <- pout$prob.test.mean
  
  residuals <- numeric(nrow(predictions))
  
  for (i in seq_len(nrow(predictions))) {
    residuals[i] <- training_cohort[[var_outcome]][i] - predictions[i, training_cohort[[grouping_var]][i]][[1]]
  }
  training_cohort$residuals <- residuals
  training_cohort_with_residuals = training_cohort
  # Train residual model
  l.out <- local_estimator_acs(
    model=model,
    Lframe = as.data.frame(training_cohort_with_residuals),
    predictors = predictors,
    outcome.var = 'residuals',
    grouping_var = grouping_var
  )
  
  # Predict for backtest cohort using global and residual models
  pout_global <- predict_global_acs(
    model=model,
    newdata=backtest_cohort,
    object=g.out,
    states=all.locs,
    predictors=predictors,
    verbose=verbose
  )
  M_global <- pout_global$prob.test.mean
  
  pout_residual <- predict_local_acs(
    model=model,
    newdata=backtest_cohort,
    object=l.out,
    states=all.locs,
    predictors=predictors,
    outcome.var = 'residuals',
    grouping_var = grouping_var,
    verbose=verbose
  )
  
  M_residual <- pout_residual$prob.test.mean
  # Combine global and residual predictions
  combined_predictions <- M_global + M_residual
  combined_predictions[combined_predictions > 1] = 1
  combined_predictions[combined_predictions < 0] = 0
  
  list(global_predictions = M_global, residual_predictions = M_residual, combined_predictions = combined_predictions, global_model_param = pout_global$model_summary_list, residual_model_param = pout_residual$model_summary_list)
}
