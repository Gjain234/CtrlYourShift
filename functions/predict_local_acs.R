predict_local_acs <- 
  function(
    model=model,
    newdata=NULL,
    object=NULL,
    states=NULL,
    predictors=c(),
    outcome.var='HAS_PUB_COV',
    grouping_var = "STATE",
    verbose=FALSE){
    library(glmnet)
    library(parallel)
    require(BART)
    require(ranger)
    require(rpart)
    require(rpart.plot)
    mc.cores = detectCores() - 2
    full.imput = TRUE
    # set up matrix to store mean predictions
    store.mean <- matrix(NA,
                         nrow=nrow(newdata),
                         ncol=length(states)
    )
    store.std <- matrix(NA,
                        nrow=nrow(newdata),
                        ncol=length(states)
    )
    
    colnames(store.mean) <- states
    colnames(store.std) <- states
    
    # list to store 1000 predictions 
    store.1000 <- list() 
    model_summary_list <- list()
    state_counts = table(newdata[[grouping_var]])
    # loop to predict
    for(i in 1:length(states)){
      state <- states[i]
      # Convert state to character for consistent comparison with Model_store names
      state_char <- as.character(state)
      existing.state <- state_char %in% names(object$Model_store)
      model_info <- list(state = state)
      
      cat("\n\n\n")
      cat("generating predictons for state:",state,"\n")
      
      
      # create test data matrix
      predict_newdata <- newdata
      predict_newdata <- data.frame(predict_newdata)
      
      # does a local model exist
      if(state_char %in% names(object$Model_store)){
        cat("-- using local model \n") 
        data = predict_newdata[,c(predictors),drop=FALSE]
        if (model == 'ranger'){
          data[is.na(data)] <- 0
          pout <- predict(object$Model_store[[state_char]],data=data)$predictions
        } else if (model == 'tree') {
          if (outcome.var != 'residuals'){
            pout <- predict(object$Model_store[[state_char]],newdata=data,type='prob')[,'1']
          } else {
            pout <- predict(object$Model_store[[state_char]],newdata=data)
          }
          if (verbose){
            model_info$ModelType <- "tree"
            model_obj = object$Model_store[[state_char]]
            model_info$NumSplits <- nrow(model_obj$frame[model_obj$frame$var != "<leaf>", ])
            model_info$FeaturesUsed <- paste(unique(model_obj$frame$var[model_obj$frame$var != "<leaf>"]), collapse = ", ")
          }
        } else if (model == 'bart') {
          available_predictors <- intersect(predictors, colnames(predict_newdata))
          x_pred <- model.matrix(
            as.formula(paste("~", paste(available_predictors, collapse = " + "), "- 1")),
            data = predict_newdata
          )
          if (outcome.var != 'residuals'){
            pout <- predict(object$Model_store[[state_char]],newdata=x_pred[,colnames(object$Model_store[[state_char]]$varcount)], mc.cores = mc.cores)$prob.test.mean
          } else {
            pout <- colMeans(predict(object$Model_store[[state_char]],newdata=x_pred[,colnames(object$Model_store[[state_char]]$varcount)], mc.cores = mc.cores))
          }
        } else if (model == 'regression') {
          available_predictors <- intersect(predictors, colnames(predict_newdata))
          x_pred <- model.matrix(
            as.formula(paste("~", paste(available_predictors, collapse = " + "), "- 1")),
            data = predict_newdata
          )
          pout <- as.numeric(predict(object$Model_store[[state_char]], newx = x_pred, type = "response"))
          if (verbose){
            coefs <- coef(object$Model_store[[state_char]])
            used_features <- rownames(coefs)[coefs[,1] != 0]
            model_info$ModelType <- "regression"
            model_info$NumFeaturesUsed <- length(used_features)
            model_info$FeaturesUsed <- paste(used_features, collapse = ", ")
          }
        }
      } else { 
        cat("-- NO MODEL FOUND \n") 
      }
      store.mean[,state]  <- pout
      model_info$NumBacktestRows <- state_counts[state]
      model_summary_list[[state]] <- model_info
      # store
      
      cat("-------------------------------------\n")
    }
    
    
    out <- list(
      prob.test.mean = store.mean,
      prob.test.std = store.std,
      model_summary_list = model_summary_list
    )
    
    return(out)
    
  }