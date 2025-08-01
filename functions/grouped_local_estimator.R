grouped_local_estimator <- function(
    model,
    Lframe, 
    predictors,
    outcome.var,
    location_weights,
    groups_to_use = NULL,
    single_group = FALSE,
    grouping_var = "STATE",
    remove_group_var = TRUE
) {
  # Load required packages
  library(dplyr)
  library(BART)
  library(ranger)
  library(rpart)
  library(rpart.plot)
  library(glmnet)
  
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
  
  # Initialize groups to use if not provided
  if (is.null(groups_to_use)) {
    groups_to_use = unique(Lframe[[grouping_var]])
  }
  
  # Get observations per group
  Lframe_group <- Lframe %>% 
    group_by(!!sym(grouping_var)) %>% 
    summarise(obs_Lframe = n()) %>% 
    arrange(obs_Lframe)
  
  weights_groups_order = colnames(location_weights)
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
    family <- if (outcome.var != 'residuals') "binomial" else "gaussian"
    cv_fit <- cv.glmnet(x_train, y_train, alpha = 1, family = family, nfolds = 10)
    optimal_lambda <- cv_fit$lambda.min
    cat("Optimal Lambda:", optimal_lambda, "\n")
    glmnet(x_train, y_train, alpha = 1, family = family, lambda = optimal_lambda)
  }
  
  # Helper function to fit tree model
  fit_tree <- function(cleaned_df, local_predictors) {
    local_formula <- reformulate(local_predictors, response = outcome.var)
    method <- if (outcome.var != 'residuals') 'class' else 'anova'
    tree_fit <- rpart(local_formula, data = cleaned_df, method = method, control = rpart.control(cp = 0))
    optimal_cp <- tree_fit$cptable[which.min(tree_fit$cptable[, "xerror"]), "CP"]
    prune(tree_fit, cp = optimal_cp)
  }
  
  # Main model fitting loop
  for (i in 1:nrow(Lframe_group)) {
    group_value <- Lframe_group[[grouping_var]][i]
    if (!(group_value %in% groups_to_use)) next
    
    cat("Fitting Local Model:", group_value, "\n")
    
    # Check if we have enough unique observations
    if (nrow(unique(Lframe[Lframe[[grouping_var]] == group_value, predictors, drop = FALSE])) <= 1) next
    
    # Get group weights and determine groups in cluster
    group_index <- if (single_group) 1 else which(weights_groups_order == group_value)
    group_weights <- location_weights[group_index,]
    groups_in_cluster <- colnames(group_weights)[as.double(group_weights) > 0]
    
    if (length(groups_in_cluster) == 0) {
      print("No weights for this location")
      next
    }
    
    # Prepare data
    subset_df <- Lframe[Lframe[[grouping_var]] %in% groups_in_cluster, , drop = FALSE]
    print(table(subset_df[[grouping_var]]))
    
    # Fit model based on type
    if (model == 'ranger') {
      data <- subset_df[, c(predictors, outcome.var), drop = FALSE]
      mout.l <- ranger(
        x = data[, predictors, drop = FALSE],
        y = data[[outcome.var]],
        classification = FALSE,
        quantreg = TRUE
      )
    } else if (model == 'tree') {
      data_prep <- prepare_data(subset_df, predictors)
      local_predictors <- setdiff(colnames(data_prep$cleaned_df), outcome.var)
      mout.l <- fit_tree(data_prep$cleaned_df, local_predictors)
    } else if (model == 'bart') {
      data_prep <- prepare_data(subset_df, predictors)
      mout.l <- if (outcome.var != 'residuals') {
        pbart(
          x.train = data_prep$x_train,
          y.train = data_prep$y_train,
          k = k,
          ntree = ntree,
          power = power,
          base = base,
          nkeeptreedraws = nkeeptreedraws
        )
      } else {
        wbart(
          x.train = data_prep$x_train,
          y.train = data_prep$y_train,
          k = k,
          ntree = ntree,
          power = power,
          base = base,
          nkeeptreedraws = nkeeptreedraws
        )
      }
    } else if (model == 'regression') {
      data_prep <- prepare_data(subset_df, predictors)
      mout.l <- fit_regression(data_prep$x_train, data_prep$y_train)
    }
    
    Model_store[[as.character(group_value)]] <- mout.l
    rm(mout.l)
    cat("Fitting Local Model: done", group_value, "\n")
  }
  
  cat("Fitting Local Models: all done\n")
  return(list(Model_store = Model_store))
}
