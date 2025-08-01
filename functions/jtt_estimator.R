source(paste0(dir,"functions/rwg_estimator.R"))
source(paste0(dir,"functions/predict_rwg.R"))

jtt_estimator <- function(
    model        = model,
    Lframe       = Lframe,
    predictors   = predictors,
    outcome.var  = outcome.var,
    grouping_var = NULL
) {
  
  ## ------------------------------------------------------------------------
  ## 1.  LOAD PACKAGES & SET PARAMS  ----------------------------------------
  ## ------------------------------------------------------------------------
  require(BART)
  require(ranger)
  require(rpart)
  require(rpart.plot)
  require(dplyr)
  require(glmnet)
  
  weights <- rep(1,nrow(Lframe))
  first_model <- rwg_estimator(
    model = model,
    Lframe = Lframe,
    predictors = predictors,
    outcome.var = outcome.var,
    weights = weights
  )
  
  if (model == 'bart'){
    pred = pnorm(first_model$Model_store$global$yhat.train.mean)
  } else if (model == 'ranger'){
    pred = first_model$Model_store$global$predictions
  } else {
    pout <- predict_rwg(
      model=model,
      newdata=Lframe,
      object=first_model,
      predictors=predictors,
      grouping_var=grouping_var,
      single_vector=TRUE
    )
    pred <- pout$prob.test.mean
  }
  y_true <- Lframe[[outcome.var]]
  error_magnitude <- abs(pred - y_true)
  weights_phase2 <- error_magnitude + 1e-4
  weights_phase2 <- weights_phase2 * (nrow(Lframe) / sum(weights_phase2))
  model_2 = rwg_estimator(
    model = model,
    Lframe = Lframe,
    predictors = predictors,
    outcome.var = outcome.var,
    weights = weights_phase2
  )
  return(model_2)
}
