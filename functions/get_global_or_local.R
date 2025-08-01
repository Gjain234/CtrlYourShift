get_global_or_local <- function(model, training_cohort, backtest_cohort, predictors, var_outcome, model_type, grouping_var) {
  library(pbapply)
  library(readr)
  library(parallel)
  all.locs <- sort(union(unique(backtest_cohort[[grouping_var]]),unique(training_cohort[[grouping_var]])))
  
  if (model_type =='global'){
    g.out <- global_estimator(
      model=model,
      Lframe=training_cohort, 
      predictors=predictors,
      outcome.var=var_outcome
    )
    M_global <- predict_global(
      model=model,
      newdata=backtest_cohort,
      object=g.out,
      predictors=predictors,
      grouping_var=grouping_var
    )
    colnames(M_global) <- all.locs
    return(M_global)
  } else {
    l.out <- local_estimator(
      model=model,
      Lframe=as.data.frame(training_cohort),
      predictors=predictors,
      outcome.var=var_outcome,
      grouping_var = grouping_var
    )
    M_local <- predict_local(
      model=model,
      newdata=backtest_cohort,
      object=l.out,
      predictors=predictors,
      outcome.var=var_outcome,
      grouping_var = grouping_var
    )
    colnames(M_local) <- all.locs
    return(M_local)
  }
}
