get_global_or_local_acs <- function(model, training_cohort, backtest_cohort, predictors, var_outcome, model_type='global', grouping_var = "STATE") {
  library(pbapply)
  library(readr)
  library(parallel)
  all.locs <- sort(union(unique(backtest_cohort[[grouping_var]]),unique(training_cohort[[grouping_var]])))
  
  if (model_type =='global'){
    g.out <- global_estimator_acs(
      model=model,
      Lframe=as.data.frame(training_cohort), 
      predictors=predictors,
      outcome.var=var_outcome
    )
    pout <- predict_global_acs(
      model=model,
      newdata=backtest_cohort,
      object=g.out,
      states=all.locs,
      predictors=predictors,
      grouping_var = grouping_var
    )
    return(pout$prob.test.mean)
  } else {
    l.out <- local_estimator_acs(
      model=model,
      Lframe=as.data.frame(training_cohort),
      predictors=predictors,
      outcome.var=var_outcome,
      grouping_var = grouping_var
    )
    pout <- predict_local_acs(
      model=model,
      newdata=backtest_cohort,
      object=l.out,
      states=all.locs,
      predictors=predictors,
      outcome.var=var_outcome,
      grouping_var = grouping_var
    )
    return(pout$prob.test.mean)
  }
}
