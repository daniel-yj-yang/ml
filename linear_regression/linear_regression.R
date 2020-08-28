# Daniel Yang, Ph.D. (daniel.yj.yang@gmail.com) @ 2020

library(datarium)
data("marketing", package = "datarium")
pairs(marketing)
cor(marketing)

y <- subset(marketing, select=c(sales))
Y <- as.matrix(y)
x <- subset(marketing, select=c(youtube, facebook, newspaper))
X <- as.matrix(x)
X_0 <- matrix( rep(1, nrow(X)), nrow(X), 1) # including the intercept term
X <- cbind(X_0, X)

# Unstandardized
theta <- solve( t(X) %*% X ) %*% t(X) %*% Y # Normal Equation
theta

model <- lm(sales ~ youtube + facebook + newspaper, data = marketing)
summary(model)

#require(ggiraphExtra)
#ggPredict(model,interactive = TRUE)

require(ggplot2)
ggplot(marketing,aes(y=sales,x=youtube))+geom_point()+geom_smooth(method="lm")
ggplot(marketing,aes(y=sales,x=facebook))+geom_point()+geom_smooth(method="lm")
ggplot(marketing,aes(y=sales,x=newspaper))+geom_point()+geom_smooth(method="lm")

# Beta (Standardized coefficients)
# remove the intercept term, and scale the X and Y
X_scaled <- subset( scale(X), select=c(youtube,facebook,newspaper)) # remove the intercept term
Y_scaled <- scale(Y)
theta_scaled <- solve( t(X_scaled) %*% X_scaled ) %*% t(X_scaled) %*% Y_scaled # Normal Equation
theta_scaled

library(QuantPsyc)
lm.beta(model)


#########################
# http://www.science.smith.edu/~jcrouser/SDS293/labs/lab10-r.html

# Ridge Regression
library(glmnet)
grid = 10^seq(10, -2, length = 100)
ridge_model = glmnet( x = X, y = Y, alpha = 0, lambda = grid)

dim(coef(ridge_model))
plot(ridge_model)    # Draw plot of coefficients

# Lasso Regression
lasso_model = glmnet( x = X, y = Y, alpha = 1, lambda = grid)
plot(lasso_model)    # Draw plot of coefficients
