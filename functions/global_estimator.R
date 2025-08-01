global_estimator <- function(
    model=model,
    Lframe=Lframe, 
    predictors=predictors,
    outcome.var=outcome.var
){
  # Default BART parameters
  k = 2
  ntree = 50
  power = 2
  base = .95
  nkeeptreedraws = 1000
  
  print("IN GLOBAL!")
  require(BART)
  require(ranger)
  require(rpart)
  require(rpart.plot)
  require(dplyr)
  require(glmnet)
  
  # Create safe names for all models
  safe_names <- make.names(names(Lframe))
  names(Lframe) <- safe_names
  predictors_safe <- make.names(predictors)
  outcome_safe <- make.names(outcome.var)
  
  # Prepare data for all models
  formula_safe <- as.formula(paste(outcome_safe, "~", paste(predictors_safe, collapse = " + "), " - 1"))
  x_train <- model.matrix(formula_safe, data = Lframe)
  y_train <- as.numeric(Lframe[[outcome_safe]])
  
  Model_store <- list()
  cat("Fitting Global Model: ", "\n")
  
  # Model-specific fitting
  if (model == 'bart') {
    mout.g <- wbart(x.train = x_train,
                y.train = y_train,
                k=k,
                power=power,
                base=base,
                nkeeptreedraws=nkeeptreedraws)
  } else if (model == 'tree') {
    tree_fit <- rpart(
      formula_safe,
      data = Lframe,
      control = rpart.control(cp = 0.001)
    )
    optimal_cp <- tree_fit$cptable[which.min(tree_fit$cptable[, "xerror"]), "CP"]
    mout.g <- prune(tree_fit, cp = optimal_cp)
  } else if (model == 'ranger') {
    mout.g <- ranger(
      x = Lframe[, predictors_safe, drop = FALSE],
      y = y_train,
      probability = FALSE
    )
  } else if (model == 'regression') {
    cv_fit <- cv.glmnet(x_train, y_train, alpha = 1, family = "gaussian", nfolds = 10)
    optimal_lambda <- cv_fit$lambda.min
    cat("Optimal Lambda:", optimal_lambda, "\n")
    mout.g <- glmnet(x_train, y_train, alpha = 1, family = "gaussian", lambda = optimal_lambda)
  }
  
  Model_store[["global"]] <- mout.g
  cat("Fitting Global Model: done", "\n")
  
  return(list(Model_store = Model_store))
}