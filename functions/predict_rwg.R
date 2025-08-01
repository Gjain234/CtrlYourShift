predict_rwg <- function(
    model,
    newdata         = NULL,
    object          = NULL,
    states          = NULL,
    predictors,
    grouping_var    = "STATE",
    verbose         = FALSE,
    single_vector   = TRUE
) {
  ## ------------------------------------------------------------------
  ## 1. LIBRARIES AND CLEAN SETUP
  ## ------------------------------------------------------------------
  library(parallel)
  require(BART)
  require(ranger)
  require(rpart)
  require(rpart.plot)
  require(glmnet)
  
  mc.cores <- max(1, detectCores() - 2)
  model_summary_list <- list()
  
  ## ------------------------------------------------------------------
  ## 2. MAKE ALL COLUMN NAMES SAFE
  ## ------------------------------------------------------------------
  all_cols       <- c(predictors, grouping_var)
  safe_lookup    <- setNames(make.names(all_cols, unique = TRUE), all_cols)
  print(names(newdata))
  print(all_cols)
  names(newdata)[match(all_cols, names(newdata))] <- safe_lookup[all_cols]
  predictors_safe <- unname(safe_lookup[predictors])
  grouping_safe   <- unname(safe_lookup[grouping_var])
  
  ## Helper: build model.matrix safely
  make_mm <- function(df, vars) {
    form <- as.formula(paste("~", paste(sprintf("`%s`", vars), collapse = " + "), "- 1"))
    model.matrix(form, data = df)
  }
  
  ## ------------------------------------------------------------------
  ## 3. SINGLE VECTOR MODE
  ## ------------------------------------------------------------------
  if (single_vector) {
    predict_newdata <- newdata
    
    if (model == "bart") {
      mdl <- object$Model_store[["global"]]
      model_vars <- colnames(mdl$varcount)
      mm <- make_mm(predict_newdata, intersect(predictors_safe, colnames(predict_newdata)))
      mm <- mm[, colnames(mm) %in% model_vars, drop = FALSE]
      pout <- colMeans(predict(mdl,
                               newdata = mm,
                               mc.cores = mc.cores))
    } else if (model == "ranger") {
      pout <- predict(object$Model_store[["global"]],
                      data = predict_newdata[, predictors_safe, drop = FALSE])$predictions
      
    } else if (model == "tree") {
      pout <- predict(object$Model_store[["global"]],
                      newdata = predict_newdata[, predictors_safe, drop = FALSE],
                      type = "prob")[, "1"]
      
    } else if (model == "regression") {
      mm <- make_mm(predict_newdata, predictors_safe)
      pout <- as.numeric(predict(object$Model_store[["global"]],
                                 newx = mm, type = "response"))
    }
    
    return(list(prob.test.mean = pout, model_summary_list = NULL))
  }
  
  ## ------------------------------------------------------------------
  ## 4. MULTI-STATE MODE
  ## ------------------------------------------------------------------
  if (is.null(states)) stop("Please supply `states` when single_vector = FALSE")
  
  predict_for_state <- function(state) {
    predict_newdata <- newdata
    mdl <- object$Model_store[["global"]]
    
    if (model == "bart") {
      model_vars <- colnames(mdl$varcount)
      mm <- make_mm(predict_newdata, predictors_safe)
      group_dummies <- grep(paste0("^", grouping_safe), colnames(mm), value = TRUE)
      mm[, group_dummies] <- 0
      current_group_dummy <- paste0(grouping_safe, state)
      if (current_group_dummy %in% colnames(mm)) {
        mm[, current_group_dummy] <- 1
      } else {
        warning("Grouping dummy ", current_group_dummy, " not found in x_pred!")
      }
      mm <- mm[, colnames(mm) %in% model_vars, drop = FALSE]
      colMeans(predict(mdl,newdata = mm, mc.cores = mc.cores))

    } else if (model == "ranger") {
      predict_newdata[[grouping_safe]] <- state
      predict(mdl, data = predict_newdata[, predictors_safe, drop = FALSE])$predictions
      
    } else if (model == "tree") {
      predict_newdata[[grouping_safe]] <- state
      predict(mdl, newdata = predict_newdata[, predictors_safe, drop = FALSE],
              type = "prob")[, "1"]
      
    } else if (model == "regression") {
      mm <- make_mm(predict_newdata, predictors_safe)
      group_dummies <- grep(paste0("^", grouping_safe), colnames(mm), value = TRUE)
      mm[, group_dummies] <- 0
      current_group_dummy <- paste0(grouping_safe, state)
      if (current_group_dummy %in% colnames(mm)) {
        mm[, current_group_dummy] <- 1
      } else {
        warning("Grouping dummy ", current_group_dummy, " not found in x_pred!")
      }
      model_vars <- rownames(coef(mdl))
      mm <- mm[, colnames(mm) %in% model_vars, drop = FALSE]
      as.numeric(predict(mdl, newx = mm, type = "response"))
    }
  }
  
  store.mean <- matrix(NA, nrow = nrow(newdata), ncol = length(states))
  colnames(store.mean) <- states
  
  for (i in seq_along(states)) {
    state <- states[i]
    cat("\nGenerating predictions for state:", state, "\n")
    pout <- predict_for_state(state)
    store.mean[, state] <- pout
    cat("-------------------------------------\n")
  }
  
  ## ------------------------------------------------------------------
  ## 5. VERBOSE MODEL SUMMARY
  ## ------------------------------------------------------------------
  if (verbose) {
    mdl <- object$Model_store[["global"]]
    if (model == "regression") {
      coefs <- coef(mdl)
      used_vars <- rownames(coefs)[coefs[, 1] != 0]
      model_summary_list$ModelType <- "regression"
      model_summary_list$NumFeaturesUsed <- length(used_vars)
      model_summary_list$FeaturesUsed <- paste(used_vars, collapse = ", ")
      
    } else if (model == "tree") {
      model_summary_list$ModelType <- "tree"
      model_summary_list$NumSplits <- nrow(mdl$frame[mdl$frame$var != "<leaf>", ])
      model_summary_list$FeaturesUsed <- paste(
        unique(mdl$frame$var[mdl$frame$var != "<leaf>"]), collapse = ", "
      )
    }
  }
  
  return(list(
    prob.test.mean = store.mean,
    model_summary_list = model_summary_list
  ))
}
