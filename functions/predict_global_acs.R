predict_global_acs <- function(
    model=model,
    newdata=NULL,
    object=NULL,
    states=NULL,
    predictors,
    grouping_var = "STATE",
    verbose=FALSE
){
  library(parallel)
  require(BART)
  require(ranger)
  require(rpart)
  require(rpart.plot)
  require(glmnet)
  mc.cores = detectCores() - 2
  full.imput = TRUE
  global_predictors = predictors
  # set up matrix to store mean predictions
  store.mean <- matrix(NA,
                       nrow=nrow(newdata),
                       ncol=length(states)
  )
  
  colnames(store.mean) <- states

  # list to store 1000 predictions 
  store.1000 <- list() 
  model_summary_list <- list()
  state_counts = table(newdata[[grouping_var]])
  # loop to predict
  for(i in 1:length(states)){
    state <- states[i]
    cat("\n\n\n")
    cat("generating predictons for state:",state,"\n")
    
    predict_newdata <- newdata
    # state_dummies <- predictors[grep(paste0("^wraps_dum_"), predictors)]
    # predict_newdata[,state_dummies] <- 0
    # predict_newdata[,paste("wraps_dum_",state,sep="")] <- 1
    # predict_newdata[,grouping_var] <- state
    predict_newdata <- data.frame(predict_newdata)
    
    cat("-- using global model \n")
    if(model == 'bart'){
      x_pred <- model.matrix(
        as.formula(paste("~", paste(global_predictors, collapse = " + "), "- 1")),
        data = predict_newdata
      )
      
      # Reset all grouping dummies
      grouping_dummy_cols <- grep(paste0("^", grouping_var), colnames(x_pred), value = TRUE)
      x_pred[, grouping_dummy_cols] <- 0
      
      # Set correct group's dummy
      current_grouping_dummy <- paste0(grouping_var, state)
      if (current_grouping_dummy %in% colnames(x_pred)) {
        x_pred[, current_grouping_dummy] <- 1
      } else {
        warning(paste("Grouping dummy", current_grouping_dummy, "not found in x_pred!"))
      }
      pout <- predict(object$Model_store[["global"]],newdata=x_pred[,colnames(object$Model_store[['global']]$varcount)], mc.cores = mc.cores)$prob.test.mean
    } else if (model == 'ranger'){
      library(ranger)
      predict_newdata[,grouping_var] <- state
      pout <- predict(object$Model_store[["global"]],data=predict_newdata[,global_predictors,drop=FALSE])$predictions
    } else if (model == 'tree'){
      predict_newdata[,grouping_var] <- state
      pout <- predict(object$Model_store[["global"]],newdata=predict_newdata[,global_predictors,drop=FALSE],type='prob')[,'1']
    } else if (model == 'regression'){
      x_pred <- model.matrix(
        as.formula(paste("~", paste(global_predictors, collapse = " + "), "- 1")),
        data = predict_newdata
      )
      
      # Reset all grouping dummies
      grouping_dummy_cols <- grep(paste0("^", grouping_var), colnames(x_pred), value = TRUE)
      x_pred[, grouping_dummy_cols] <- 0
      
      # Set correct group's dummy
      current_grouping_dummy <- paste0(grouping_var, state)
      if (current_grouping_dummy %in% colnames(x_pred)) {
        x_pred[, current_grouping_dummy] <- 1
      } else {
        warning(paste("Grouping dummy", current_grouping_dummy, "not found in x_pred!"))
      }
      pout <- as.numeric(predict(object$Model_store[["global"]],newx=x_pred, type = "response"))
    }
    store.mean[,state]  <- pout
    rm(pout)
    cat("-------------------------------------\n")
  }
  if (verbose){
    if (model == 'regression'){
      coefs <- coef(object$Model_store[["global"]])
      used_features <- rownames(coefs)[coefs[,1] != 0]
      model_summary_list$ModelType <- "regression"
      model_summary_list$NumFeaturesUsed <- length(used_features)
      model_summary_list$FeaturesUsed <- paste(used_features, collapse = ", ")
    } else if (model == 'tree'){
      model_summary_list$ModelType <- "tree"
      model_obj = object$Model_store[["global"]]
      model_summary_list$NumSplits <- nrow(model_obj$frame[model_obj$frame$var != "<leaf>", ])
      model_summary_list$FeaturesUsed <- paste(unique(model_obj$frame$var[model_obj$frame$var != "<leaf>"]), collapse = ", ")
    }
  }
  out <- list(
    prob.test.mean = store.mean,
    model_summary_list = model_summary_list
  )
  
  return(out)
  
}
