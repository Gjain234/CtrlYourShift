predict_global <- function(
    model,
    newdata = NULL,
    object = NULL,
    predictors,
    grouping_var,
    single_vector=FALSE) {
  library(parallel)
  require(BART)
  require(ranger)
  require(rpart)
  require(rpart.plot)
  require(glmnet)
  
  mc.cores <- detectCores() - 2
  global_predictors <- predictors
  
  # Ensure column names are safe
  safe_names <- make.names(names(newdata))
  names(newdata) <- safe_names
  grouping_var <- make.names(grouping_var)
  predictors_safe <- make.names(predictors)
  
  # Model matrix (used for BART and regression)
  formula_safe <- as.formula(paste("~", paste(predictors_safe, collapse = " + "), "- 1"))
  if (single_vector) {
    x_pred <- model.matrix(formula_safe, data = newdata)
  
    if (model == 'bart') {
      model_vars = colnames(object$Model_store[['global']]$varcount)
      x_pred <- x_pred[, colnames(x_pred) %in% model_vars]
      pout <- colMeans(predict(object$Model_store[["global"]],
                      newdata = x_pred,
                      mc.cores = mc.cores))
      
    } else if (model == 'ranger') {
      pout <- predict(object$Model_store[["global"]],
                      data = newdata[, global_predictors, drop = FALSE])$predictions
      
    } else if (model == 'tree') {
      pout <- predict(object$Model_store[["global"]],
                      newdata = newdata[, global_predictors, drop = FALSE])
    } else if (model == 'regression') {
      pout <- as.numeric(predict(object$Model_store[["global"]],
                                newx = x_pred, type = "response"))
    }
    
    return(pout)
  }
  # Unique values of the group
  unique_groups <- sort(unique(newdata[[grouping_var]]))
  n <- nrow(newdata)
  g <- length(unique_groups)
  
  # Placeholder for the matrix of predictions
  pred_matrix <- matrix(NA, nrow = n, ncol = g)
  colnames(pred_matrix) <- unique_groups
  
  for (i in seq_along(unique_groups)) {
    group_val <- unique_groups[i]
    
    # Clone the data and set everyone to this group
    newdata_group <- newdata
    
    # Predict
    if (model == 'bart') {
      x_pred <- model.matrix(formula_safe, data = newdata_group)
      group_dummies <- grep(paste0("^", grouping_var), colnames(x_pred), value = TRUE)
      x_pred[, group_dummies] <- 0
      current_group_dummy <- paste0(grouping_var, group_val)
      if (current_group_dummy %in% colnames(x_pred)) {
        x_pred[, current_group_dummy] <- 1
      } else {
        warning("Grouping dummy ", current_group_dummy, " not found in x_pred!")
      }
      model_vars <- colnames(object$Model_store[['global']]$varcount)
      x_pred <- x_pred[, colnames(x_pred) %in% model_vars, drop = FALSE]
      pout <- colMeans(predict(object$Model_store[["global"]],
                      newdata = x_pred,
                      mc.cores = mc.cores))
      
    } else if (model == 'ranger') {
      newdata_group[[grouping_var]] <- group_val
      pout <- predict(object$Model_store[["global"]],
                      data = newdata_group[, global_predictors, drop = FALSE])$predictions
      
    } else if (model == 'tree') {
      newdata_group[[grouping_var]] <- group_val
      pout <- predict(object$Model_store[["global"]],
                      newdata = newdata_group[, global_predictors, drop = FALSE])
      
    } else if (model == 'regression') {
      x_pred <- model.matrix(formula_safe, data = newdata_group)
      group_dummies <- grep(paste0("^", grouping_var), colnames(x_pred), value = TRUE)
      x_pred[, group_dummies] <- 0
      current_group_dummy <- paste0(grouping_var, group_val)
      if (current_group_dummy %in% colnames(x_pred)) {
        x_pred[, current_group_dummy] <- 1
      } else {
        warning("Grouping dummy ", current_group_dummy, " not found in x_pred!")
      }
      model_vars <- rownames(coef(object$Model_store[["global"]]))
      x_pred <- x_pred[, colnames(x_pred) %in% model_vars, drop = FALSE]
      pout <- as.numeric(predict(object$Model_store[["global"]],
                                 newx = x_pred, type = "response"))
    }
    
    pred_matrix[, i] <- pout
  }
  
  return(pred_matrix)
}
