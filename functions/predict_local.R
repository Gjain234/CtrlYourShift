predict_local <- function(
  model,
  newdata,
  object,
  predictors,
  outcome.var,
  grouping_var,
  remove_group_var=TRUE,
  single_vector=FALSE
) {
  library(glmnet)
  library(parallel)
  require(BART)
  require(ranger)
  require(rpart)
  require(rpart.plot)
  mc.cores <- detectCores() - 2

  if (remove_group_var) {
    predictors <- setdiff(predictors, grouping_var)
  }

  # Create safe names and formula
  safe_names <- make.names(names(newdata))
  names(newdata) <- safe_names
  predictors_safe <- make.names(predictors)
  formula_safe <- as.formula(paste("~", paste(predictors_safe, collapse = " + "), "- 1"))
  print(head(newdata))
  print(nrow(newdata))
  print(formula_safe)
  x_pred <- model.matrix(formula_safe, data = newdata)

  if (single_vector) {
    preds <- rep(NA, nrow(newdata))
    for (state in unique(newdata[[grouping_var]])) {
      state_char <- as.character(state)
      row_indices <- which(newdata[[grouping_var]] == state_char)
      predict_newdata <- newdata[row_indices, , drop = FALSE]

      if (state_char %in% names(object$Model_store)) {
        if (model == 'ranger') {
          data <- predict_newdata[, predictors, drop = FALSE]
          data[is.na(data)] <- 0
          pout <- predict(object$Model_store[[state_char]], data = data)$predictions
        } else if (model == 'tree') {
          data <- predict_newdata[, predictors, drop = FALSE]
          pout <- predict(object$Model_store[[state_char]], newdata = data)
        } else if (model == 'bart') {
          model_vars = colnames(object$Model_store[[state_char]]$varcount)
          x_pred_state <- x_pred[row_indices, , drop = FALSE]
          x_pred_state <- x_pred_state[, colnames(x_pred_state) %in% model_vars]
          pout <- colMeans(predict(object$Model_store[[state_char]],
                             newdata = x_pred_state,
                             mc.cores = mc.cores))
        } else if (model == 'regression') {
          coefs <- coef(object$Model_store[[state_char]])
          model_vars <- rownames(coefs)
          x_pred_state <- x_pred[row_indices, , drop = FALSE]
          x_pred_state <- x_pred_state[, colnames(x_pred_state) %in% model_vars]
          pout <- as.numeric(predict(object$Model_store[[state_char]], newx = x_pred_state, type = "response"))
        }

        preds[row_indices] <- pout
      } else {
        warning(paste("No model found for state:", state_char))
      }
    }
    return(preds)
  } else {
    # Return matrix of predictions where each column is predictions from one group's model
    all_states <- sort(unique(newdata[[grouping_var]]))
    M_local <- matrix(NA, nrow = nrow(newdata), ncol = length(all_states))
    colnames(M_local) <- all_states

    # For each group's model, predict for all rows
    for (state in all_states) {
      state_char <- as.character(state)
      print(paste("Predicting for group:", state_char))
      if (state_char %in% names(object$Model_store)) {
        if (model == 'ranger') {
          data <- newdata[, predictors, drop = FALSE]
          data[is.na(data)] <- 0
          pout <- predict(object$Model_store[[state_char]], data = data)$predictions
        } else if (model == 'tree') {
          data <- newdata[, predictors, drop = FALSE]
          pout <- predict(object$Model_store[[state_char]], newdata = data)
        } else if (model == 'bart') {
          model_vars = colnames(object$Model_store[[state_char]]$varcount)
          x_pred_state <- x_pred[, colnames(x_pred) %in% model_vars]
          pout <- colMeans(predict(object$Model_store[[state_char]],
                             newdata = x_pred_state,
                             mc.cores = mc.cores))
        } else if (model == 'regression') {
          coefs <- coef(object$Model_store[[state_char]])
          model_vars <- rownames(coefs)
          x_pred_state <- x_pred[, colnames(x_pred) %in% model_vars]
          pout <- as.numeric(predict(object$Model_store[[state_char]], newx = x_pred_state, type = "response"))
        }

        M_local[, state_char] <- pout
      } else {
        warning(paste("No model found for state:", state_char))
      }
    }
    return(M_local)
  }
}
