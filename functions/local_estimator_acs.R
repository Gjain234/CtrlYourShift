local_estimator_acs <- function(
    model=model,
    Lframe=Lframe, 
    predictors=predictors,
    outcome.var=outcome.var,
    grouping_var = "STATE"
){
  require(BART)
  require(ranger)
  require(rpart)
  require(rpart.plot)
  require(dplyr)
  require(glmnet)
  k = 2 # pbart default is 2
  ntree = 50 # pbart default us 50
  power = 2 # pbart default is 2
  base  = .95 # pbart default is .95
  nkeeptreedraws=1000
  Model_store <- list()  
  # which affiliates are we trying to model training obs per location
  Lframe_state <- Lframe %>% group_by(!!sym(grouping_var)) %>% summarise(obs_Lframe=n()) %>% arrange(obs_Lframe)
  
  # fit local models
  invalirs<-FALSE
  for(i in 1:nrow(Lframe_state)){
    # Convert group value to character for consistent storage in Model_store
    group_value <- Lframe_state[[grouping_var]][i]
    group_value_char <- as.character(group_value)
    
    cat("Fitting Local Model:", as.character(group_value) ,"\n")  
    if (nrow(unique(Lframe[Lframe[[grouping_var]] == group_value,predictors,drop=FALSE]))>1){
      subset_df <- Lframe[Lframe[[grouping_var]] == group_value, , drop = FALSE]
      single_value_columns <- function(df) {
        sapply(df, function(col) length(unique(col)) == 1)
      }
      cleaned_df <- subset_df[,c(predictors,outcome.var),drop=FALSE]
      cols_to_remove <- single_value_columns(cleaned_df)
      cols_to_remove[names(cleaned_df) %in% outcome.var] <- FALSE
      cleaned_df <- cleaned_df[, !cols_to_remove, drop = FALSE]
      if (model == 'bart'){
        available_predictors <- intersect(predictors, colnames(subset_df))
        x_train <- model.matrix(
          as.formula(paste("~", paste(available_predictors, collapse = " + "), "- 1")),
          data = subset_df
        )
        if (outcome.var =='residuals'){
          mout.l <-  wbart(x.train = x_train,
                           y.train = subset_df[,outcome.var],
                           k=k,
                           power=power,
                           base=base,
                           nkeeptreedraws=nkeeptreedraws)
        } else {
          mout.l <-  pbart(x.train = x_train,
                           y.train = subset_df[,outcome.var],
                           k=k,
                           ntree=ntree,
                           power=power,
                           base=base,
                           nkeeptreedraws=nkeeptreedraws)
        }
      } else if (model == 'regression') {
        available_predictors <- intersect(predictors, colnames(subset_df))
        x_train <- model.matrix(
          as.formula(paste("~", paste(available_predictors, collapse = " + "), "- 1")),
          data = subset_df
        )
        if (outcome.var != 'residuals') {
          y_train = as.numeric(subset_df[[outcome.var]])
          class_counts <- table(y_train)
          # Check that both classes have at least 10 observations
          if (all(c(0, 1) %in% names(class_counts)) && all(class_counts >= 10)) {
            cv_fit <- cv.glmnet(x_train, y_train, alpha = 1, family = "binomial", nfolds = 10)
            optimal_lambda <- cv_fit$lambda.min
          } else {
            cat("Insufficient class variation in y — using default lambda = 0.01...\n")
            optimal_lambda <- 0.01
          }
          cat("Optimal Lambda:", optimal_lambda, "\n")
          mout.l <- glmnet(x_train, y_train, alpha = 1, family = "binomial", lambda = optimal_lambda)
        } else {
          y_train = subset_df$residuals
          cv_fit <- cv.glmnet(x_train, y_train, alpha = 1, family = "gaussian", nfolds = 10)
          optimal_lambda <- cv_fit$lambda.min
          cat("Optimal Lambda:", optimal_lambda, "\n")
          mout.l <- glmnet(x_train, y_train, alpha = 1, family = "gaussian", lambda = optimal_lambda)
        }
      } else {
        local_predictors <- setdiff(colnames(cleaned_df), outcome.var)
        local_formula = reformulate(local_predictors, response = outcome.var)
        if (model == 'ranger'){
          mout.l <- ranger(local_formula, data=as.data.frame(cleaned_df),classification=FALSE,quantreg = TRUE)
        } else if (model == 'tree'){
          if (outcome.var != 'residuals'){
            tree_fit <- rpart(local_formula, data = cleaned_df,method='class',control = rpart.control(cp = 0))
            optimal_cp <- tree_fit$cptable[which.min(tree_fit$cptable[, "xerror"]), "CP"]
            mout.l <- prune(tree_fit, cp = optimal_cp)
          } else {
            tree_fit <- rpart(local_formula, data = cleaned_df,control = rpart.control(cp = 0))
            optimal_cp <- tree_fit$cptable[which.min(tree_fit$cptable[, "xerror"]), "CP"]
            mout.l <- prune(tree_fit, cp = optimal_cp)
          }
        }
      }
      Model_store[[group_value_char]] <- mout.l
      rm(mout.l)
      cat("Fitting Local Model: done", group_value ,"\n")
    }
  }
  cat("Fitting Local Models: all done \n")
  out <- list(
    Model_store=Model_store
  )
  return(out)
}