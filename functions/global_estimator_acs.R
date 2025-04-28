global_estimator_acs <- function(
    model=model,
    Lframe=Lframe, 
    predictors=predictors,
    outcome.var=outcome.var
){
  k = 2 # pbart default is 2
  ntree = 50 # pbart default us 50
  power = 2 # pbart default is 2
  base  = .95 # pbart default is .95
  nkeeptreedraws=1000
  print("IN GLOBAL!")
  require(BART)
  require(ranger)
  require(rpart)
  require(rpart.plot)
  require(dplyr)
  require(glmnet)
  Model_store <- list()  
  
  # global model
  cat("Fitting Gobal Model: ","\n") 
  if (model == 'bart'){
    x_train <- model.matrix(as.formula(paste("~", paste(predictors, collapse = " + "), "- 1")), data = Lframe)
    mout.g <-  pbart(x.train = x_train,
                     y.train = Lframe[,outcome.var],
                     k=k,
                     ntree=ntree,
                     power=power,
                     base=base,
                     nkeeptreedraws=nkeeptreedraws)
  } else if (model == 'tree'){
    tree_fit <- rpart(reformulate(predictors, response = outcome.var), data = Lframe[,c(outcome.var,predictors),drop=FALSE],method='class',control = rpart.control(cp = 0.001))
    optimal_cp <- tree_fit$cptable[which.min(tree_fit$cptable[, "xerror"]), "CP"]
    mout.g <- prune(tree_fit, cp = optimal_cp)
  } else if (model == 'ranger'){
    mout.g <- ranger(reformulate(predictors, response = outcome.var), data = Lframe[,c(predictors,outcome.var),drop=FALSE],probability=FALSE)
  } else if (model == 'regression'){
    x_train <- model.matrix(as.formula(paste("~", paste(predictors, collapse = " + "), "- 1")), data = Lframe)
    y_train <- as.numeric(Lframe[[outcome.var]])
    cv_fit <- cv.glmnet(x_train, y_train, alpha = 1, family = "binomial", nfolds = 10)
    optimal_lambda <- cv_fit$lambda.min
    cat("Optimal Lambda:", optimal_lambda, "\n")
    mout.g <- glmnet(x_train, y_train, alpha = 1, family = "binomial", lambda = optimal_lambda)
  }
  Model_store[["global"]] <- mout.g
  cat("Fitting Gobal Model: done","\n")  
  out <- list(
    Model_store=Model_store
  )
  
  return(out)
  
}