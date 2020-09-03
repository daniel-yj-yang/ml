rm(list=ls())
cat("\014")

# https://www.datacamp.com/community/tutorials/tutorial-ridge-lasso-elastic-net
# https://www.r-bloggers.com/ridge-regression-and-the-lasso/
# https://www.pluralsight.com/guides/linear-lasso-and-ridge-regression-with-r
# https://towardsdatascience.com/ridge-and-lasso-regression-a-complete-guide-with-python-scikit-learn-e20e34bcbf0b

# Initialization

eval_results <- function(y, y_pred) {
  SSE <- sum((y_pred - y)^2)
  RMSE = sqrt(SSE/length(y))
  
  SST <- sum((y - mean(y))^2)
  R_square <- 1 - (SSE / SST)
  
  # Model performance metrics
  data.frame(
    RMSE = RMSE,
    R_square = R_square
  )
}

library(glmnet)  # for ridge regression
library(MASS)    # for ridge regression
library(data.table)

set.seed(123)    # seed for reproducibility

swiss <- datasets::swiss
#X <- subset(swiss, select = -c( Fertility ))
X <- model.matrix(Fertility~., swiss)[,-1] # to avoid the following error message in ridge regression:
#Error in elnet(x, is.sparse, ix, jx, y, weights, offset, type.gaussian,  : 
#                 'list' object cannot be coerced to type 'double'
y <- subset(swiss, select = c("Fertility"))

#train_idx = sample(1:nrow(X), nrow(X)/2)
train_idx = c(31, 15, 14, 3, 42, 37, 45, 25, 26, 27, 5, 38, 28, 9, 29, 44, 8, 39, 7, 10, 34, 19, 4)
test_idx = (-train_idx)

train <- swiss[train_idx,]
X_train <- X[train_idx,]
y_train <- y[train_idx,]

test <- swiss[test_idx,]
X_test <- X[test_idx,]
y_test = y[test_idx,]

############################################################################################
# 1. Linear Regression - OLS

lr_model <- lm(Fertility~., data = train)
summary(lr_model)

# Prediction and evaluation on train data
y_train_pred_lr <- predict(lr_model, newdata = train)
eval_results(y_train, y_train_pred_lr)

# Prediction and evaluation on test data
y_test_pred_lr <- predict(lr_model, newdata = test)
eval_results(y_test, y_test_pred_lr)

############################################################################################
# 2. Ridge Regression
# http://www.science.smith.edu/~jcrouser/SDS293/labs/lab10-r.html

rr_insights <- function(ridge.model, nth) {
  print( paste0('The ', nth, '-th lambda value = ', ridge.model$lambda[nth] ) )
  print( paste0('The coefficients associated with the ', nth, '-th lambda value' ) )
  print( coef(ridge.model)[,nth] )
  print( paste0('The L2 norm associated with the ', nth, '-th lambda value = ', sqrt(sum(coef(ridge.model)[-1,nth]^2)) ) )
}

plot_L2_norm_vs_lambda <- function(ridge.model) {
  n_lambdas <- ncol(coef(ridge.model))
  L2_norm_vs_lambda_dt <- data.table( L2_norm = 123,
                                   lambda = 123)[0]
  for(ith in 1:n_lambdas) {
    this_L2_norm  <- sqrt(sum(coef(ridge.model)[-1,ith]^2))
    this_lambda <- ridge.model$lambda[ith]
    L2_norm_vs_lambda_dt <- rbind(L2_norm_vs_lambda_dt,
                                  data.table( L2_norm = this_L2_norm,
                                              lambda = this_lambda))
  }
  
  require(ggplot2)
  ggplot(L2_norm_vs_lambda_dt) + 
    aes(x = log(L2_norm_vs_lambda_dt$lambda), y = L2_norm_vs_lambda_dt$L2_norm, color = "red") + 
    geom_line(size=2) +
    #ggeom_smooth(method = 'loess') +
    xlab( 'log(λ)' ) +
    ylab( 'L2 norm' ) +
    ggtitle("L2 norm vs log(λ)")
}

pairs(X)

#################################################################################
#### glmnet
# set up the initial model without the best lambda yet
# http://www.science.smith.edu/~jcrouser/SDS293/labs/lab10-r.html

lambdas <- 10^seq(5, -3, length = 300)
ridge_reg_model <- glmnet(x = X_train, y = y_train, alpha = 0, lambda = lambdas, thresh = 1e-12)

rr_insights(ridge_reg_model, 226)
dim(coef(ridge_reg_model))
plot(ridge_reg_model, xvar = "norm",   label = T)
plot(ridge_reg_model, xvar = "lambda", label = T)
plot(ridge_reg_model, xvar = "dev",    label = T)
plot_L2_norm_vs_lambda(ridge_reg_model)

summary(ridge_reg_model)

# find the best lambda
set.seed(123)    # seed for reproducibility
cv.ridge <- cv.glmnet( x = X_train, y = y_train, alpha = 0, lambda = lambdas, nfolds = 10 )
plot(cv.ridge)
best_lambda <- cv.ridge$lambda.min
print(paste0('best_lambda (empiricially derived): ', best_lambda))
print(predict(ridge_reg_model, s = best_lambda, type = "coefficients"))

# Prediction and evaluation on train data -- equivalent to OLS
y_train_pred_rr <- predict(ridge_reg_model, s = 0, newx = X_train, exact = T, x = X_train, y = y_train)
eval_results(y_train, y_train_pred_rr)

# Prediction and evaluation on test data -- equivalent to OLS
y_test_pred_rr <- predict(ridge_reg_model, s = 0, newx = X_test, exact = T, x = X_train, y = y_train)
eval_results(y_test, y_test_pred_rr)

# Prediction and evaluation on train data -- Ridge Regression
y_train_pred_rr <- predict(ridge_reg_model, s = best_lambda, newx = X_train)
eval_results(y_train, y_train_pred_rr)

# Prediction and evaluation on test data -- Ridge Regression
y_test_pred_rr <- predict(ridge_reg_model, s = best_lambda, newx = X_test)
eval_results(y_test, y_test_pred_rr)

#################################################################################
#### MASS
ridge_reg_model_2 <- lm.ridge( Fertility ~ ., train, lambda = lambdas )
#print(ridge_reg_model_2)
#plot(ridge_reg_model_2)
select(ridge_reg_model_2) # another way to obtain best lambda
print('smallest value of GCV is another empirically derived index of best lambda')


############################################################################################
# 3. Lasso Regression
## http://www.science.smith.edu/~jcrouser/SDS293/labs/lab10-r.html
## https://www.r-bloggers.com/ridge-regression-and-the-lasso/
## https://www.r-bloggers.com/ordinary-least-squares-ols-linear-regression-in-r/
## https://www.analyticsvidhya.com/blog/2016/01/ridge-lasso-regression-python-complete-tutorial/#four
## https://www.datacamp.com/community/tutorials/tutorial-ridge-lasso-elastic-net
## https://courses.analyticsvidhya.com/courses/take/big-mart-sales-prediction-using-r/texts/6120184-model-building
## https://rstudio-pubs-static.s3.amazonaws.com/381886_981132516a8e437284327a405ca4d91a.html

