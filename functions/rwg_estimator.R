rwg_estimator <- function(
    model        = model,
    Lframe       = Lframe,
    predictors   = predictors,
    outcome.var  = outcome.var,
    weights = NULL
) {
  
  ## ------------------------------------------------------------------------
  ## 1.  LOAD PACKAGES & SET PARAMS  ----------------------------------------
  ## ------------------------------------------------------------------------
  cat("IN GLOBAL!\n")
  require(BART)
  require(ranger)
  require(rpart)
  require(rpart.plot)
  require(dplyr)
  require(glmnet)
  
  k                 <- 2
  ntree             <- 50
  power             <- 2
  base              <- 0.95
  nkeeptreedraws    <- 1000
  Model_store       <- list()
  
  ## ------------------------------------------------------------------------
  ## 2.  CLEAN COLUMN NAMES SAFELY  -----------------------------------------
  ## ------------------------------------------------------------------------
  all_cols       <- c(predictors, outcome.var)
  names(Lframe) <- make.names(names(Lframe), unique = TRUE)
  if (is.null(weights)) {
    weights <- rep(1,nrow(Lframe))
  }
  # Lookup safe versions of the vars we care about
  safe_lookup    <- setNames(names(Lframe), names(Lframe))  # update if needed
  predictors_safe <- make.names(predictors)
  outcome_safe    <- make.names(outcome.var)
  
  ## Build formula once (intercept removed)
  rhs             <- paste(sprintf("`%s`", predictors_safe), collapse = " + ")
  formula_safe    <- as.formula(paste0("`", outcome_safe, "` ~ ", rhs, " - 1"))
  
  ## ------------------------------------------------------------------------
  ## 3.  MODEL FITTING BLOCK  -----------------------------------------------
  ## ------------------------------------------------------------------------
  cat("Fitting Global Model...\n")
  
  if (model == "bart") {
    x_train <- model.matrix(formula_safe, data = Lframe)
    w_bart = sqrt(1 / weights)
    w_bart <- w_bart * (nrow(Lframe) / sum(w_bart))
    mout.g <- wbart(x.train = x_train,
                    y.train = Lframe[[outcome_safe]],
                    k = k,
                    power = power,
                    base = base,
                    w = w_bart,
                    nkeeptreedraws = nkeeptreedraws)
    
  } else if (model == "tree") {
    # Build full formula with intercept for rpart
    tree_formula <- as.formula(
      paste0("`", outcome_safe, "` ~ ", paste(sprintf("`%s`", predictors_safe), collapse = " + "))
    )
    
    tree_fit <- rpart(tree_formula, data = Lframe,
                      method = "class",
                      control = rpart.control(cp = 0.001),weights = weights)
    
    optimal_cp <- tree_fit$cptable[which.min(tree_fit$cptable[, "xerror"]), "CP"]
    mout.g     <- prune(tree_fit, cp = optimal_cp)
    
  } else if (model == "ranger") {
    x_data <- Lframe[, predictors_safe, drop = FALSE]
    y_data <- Lframe[[outcome_safe]]
    
    mout.g <- ranger(x = x_data, y = y_data,
                     classification = (outcome.var == "has_chronic_condition"),
                     probability = FALSE, case.weights = weights)
    
  } else if (model == "regression") {
    x_train <- model.matrix(formula_safe, data = Lframe)
    y_train <- as.numeric(Lframe[[outcome_safe]])
    
    cv_fit <- cv.glmnet(x_train, y_train, alpha = 1, family = "binomial", nfolds = 10,weights = weights)
    optimal_lambda <- cv_fit$lambda.min
    cat("Optimal Lambda:", optimal_lambda, "\n")
    
    mout.g <- glmnet(x_train, y_train, alpha = 1, family = "binomial", lambda = optimal_lambda)
  }
  ## ------------------------------------------------------------------------
  ## 4.  RETURN OUTPUT  -----------------------------------------------------
  ## ------------------------------------------------------------------------
  Model_store[["global"]] <- mout.g
  cat("Fitting Global Model: done\n")
  
  return(list(Model_store = Model_store))
}
