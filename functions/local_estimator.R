local_estimator <- function(
    model = model,
    Lframe = Lframe, 
    predictors = predictors,
    outcome.var = outcome.var,
    grouping_var = "STATE",
    remove_group_var = TRUE
) {
  # Load required packages
  require(BART)
  require(ranger)
  require(rpart)
  require(rpart.plot)
  require(dplyr)
  require(glmnet)
  
  # Default BART parameters
  k = 2
  ntree = 50
  power = 2
  base = .95
  nkeeptreedraws = 1000
  
  # Remove grouping_var from predictors if specified
  if (remove_group_var) {
    predictors <- setdiff(predictors, grouping_var)
  }
  
  # Get observations per location
  Lframe_state <- Lframe %>% 
    group_by(!!sym(grouping_var)) %>% 
    summarise(obs_Lframe = n()) %>% 
    arrange(obs_Lframe)
  
  Model_store <- list()
  
  # Helper function to prepare data for modeling
  prepare_data <- function(subset_df, available_predictors) {
    # Remove columns with single values
    single_value_columns <- function(df) {
      sapply(df, function(col) length(unique(col)) == 1)
    }
    
    cleaned_df <- subset_df[, c(available_predictors, outcome.var), drop = FALSE]
    cols_to_remove <- single_value_columns(cleaned_df)
    cols_to_remove[names(cleaned_df) %in% outcome.var] <- FALSE
    cleaned_df <- cleaned_df[, !cols_to_remove, drop = FALSE]
    
    # Create formula and model matrix
    formula_safe <- as.formula(paste("~", paste(sprintf("`%s`", colnames(cleaned_df)[-ncol(cleaned_df)]), collapse = " + "), "- 1"))
    x_train <- model.matrix(formula_safe, data = cleaned_df)
    y_train <- cleaned_df[[outcome.var]]
    
    list(cleaned_df = cleaned_df, x_train = x_train, y_train = y_train)
  }
  
  # Helper function to fit regression model
  fit_regression <- function(x_train, y_train) {
    cv_fit <- cv.glmnet(x_train, y_train, alpha = 1, family = "gaussian", nfolds = 10)
    optimal_lambda <- cv_fit$lambda.min
    cat("Optimal Lambda:", optimal_lambda, "\n")
    glmnet(x_train, y_train, alpha = 1, family = "gaussian", lambda = optimal_lambda)
  }
  
  # Helper function to fit tree model
  fit_tree <- function(cleaned_df, local_predictors) {
    # Create formula
    local_formula <- reformulate(local_predictors, response = outcome.var)
    
    # Fit and prune tree
    method <- 'anova'
    tree_fit <- rpart(local_formula, data = cleaned_df, method = method, control = rpart.control(cp = 0))
    optimal_cp <- tree_fit$cptable[which.min(tree_fit$cptable[, "xerror"]), "CP"]
    prune(tree_fit, cp = optimal_cp)
  }
  
  # Main model fitting loop
  for (i in 1:nrow(Lframe_state)) {
    group_value <- as.character(Lframe_state[[grouping_var]][i])
    cat("Fitting Local Model:", group_value, "\n")
    
    subset_df <- Lframe[Lframe[[grouping_var]] == group_value, , drop = FALSE]
    if (nrow(unique(subset_df[, predictors, drop = FALSE])) <= 1) next
    
    # Prepare data
    data_prep <- prepare_data(subset_df, predictors)
    cleaned_df <- data_prep$cleaned_df
    x_train <- data_prep$x_train
    y_train <- data_prep$y_train
    
    # Fit model based on type
    if (model == 'bart') {
      mout.l <- wbart(x.train = x_train, y.train = y_train, k = k, power = power, base = base, nkeeptreedraws = nkeeptreedraws)
    } else if (model == 'regression') {
      mout.l <- fit_regression(x_train, y_train)
    } else if (model == 'ranger') {
      local_predictors <- setdiff(colnames(cleaned_df), outcome.var)
      mout.l <- ranger(
        x = cleaned_df[, local_predictors, drop = FALSE],
        y = y_train,
        classification = FALSE,
        quantreg = TRUE
      )
    } else if (model == 'tree') {
      local_predictors <- setdiff(colnames(cleaned_df), outcome.var)
      mout.l <- fit_tree(cleaned_df, local_predictors)
    }
    
    Model_store[[group_value]] <- mout.l
    rm(mout.l)
    cat("Fitting Local Model: done", group_value, "\n")
  }
  
  cat("Fitting Local Models: all done\n")
  return(list(Model_store = Model_store))
}