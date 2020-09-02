#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 21:48:35 2020

@author: daniel
"""

# https://www.datacamp.com/community/tutorials/tutorial-ridge-lasso-elastic-net
# https://www.r-bloggers.com/ridge-regression-and-the-lasso/
# https://www.pluralsight.com/guides/linear-lasso-and-ridge-regression-with-r
# https://towardsdatascience.com/ridge-and-lasso-regression-a-complete-guide-with-python-scikit-learn-e20e34bcbf0b

# Initialization

import math
import statistics
import random
import pandas as pd

def eval_results(y, y_pred):
    SSE = sum((y_pred-y)**2)
    RMSE = math.sqrt(SSE/len(y))
    SST = sum((y-statistics.mean(y))**2)
    R_square = 1 - (SSE / SST)
    print( "RMSE = {:.06f}, R_square = {:.06f}".format(RMSE, R_square) )

random.seed(123)

data = pd.read_csv('/Users/daniel/Data-Science/Data/from_R_datasets/swiss/swiss.csv')

X = data.drop(["Fertility"], axis=1)
y = data["Fertility"]

train_idx = [31, 15, 14, 3, 42, 37, 45, 25, 26, 27, 5, 38, 28, 9, 29, 44, 8, 39, 7, 10, 34, 19, 4]
train_idx = [x - 1 for x in train_idx]

X_train = X.loc[train_idx]
y_train = y.loc[train_idx]

X_test = X.drop(train_idx, axis=0)
y_test = y.drop(train_idx, axis=0)

import statsmodels.api as sm
X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)


# 1. Linear Regression - OLS
print('\n--------------------------\n2. Linear Regression\n--------------------------')
OLS_model = sm.OLS(y_train,X_train_sm)
linear_reg_model_1 = OLS_model.fit()
print('\nstatsmodels')
print(linear_reg_model_1.params)
y_train_pred_linearreg_1 = linear_reg_model_1.predict(X_train_sm)
y_test_pred_linearreg_1  = linear_reg_model_1.predict(X_test_sm)
eval_results(y_train, y_train_pred_linearreg_1)
eval_results(y_test,  y_test_pred_linearreg_1 )


from sklearn.linear_model import LinearRegression
linear_reg_model_2 = LinearRegression()
linear_reg_model_2.fit(X_train,y_train)
print('\nscikit-learn')
print(linear_reg_model_2.intercept_, linear_reg_model_2.coef_)
y_train_pred_linearreg_2 = linear_reg_model_2.predict(X_train)
y_test_pred_linearreg_2  = linear_reg_model_2.predict(X_test)
eval_results(y_train, y_train_pred_linearreg_2)
eval_results(y_test,  y_test_pred_linearreg_2 )




# 2. Ridge Regression
print('\n--------------------------\n2. Ridge Regression\n--------------------------')
OLS_model = sm.OLS(y_train,X_train_sm)
ridge_reg_model_1 = OLS_model.fit_regularized(L1_wt = 0.0, alpha = 100)
print('\nstatsmodels')
print(ridge_reg_model_1.params)
y_train_pred_ridgereg_1 = ridge_reg_model_1.predict(X_train_sm)
y_test_pred_ridgereg_1  = ridge_reg_model_1.predict(X_test_sm)
eval_results(y_train, y_train_pred_ridgereg_1)
eval_results(y_test,  y_test_pred_ridgereg_1 )


from sklearn.linear_model import Ridge
ridge_reg_model_2 = Ridge(alpha = 100)
ridge_reg_model_2.fit(X_train,y_train)
print('\nscikit-learn')
print(ridge_reg_model_2.intercept_, ridge_reg_model_2.coef_)
y_train_pred_ridgereg_2 = ridge_reg_model_2.predict(X_train)
y_test_pred_ridgereg_2  = ridge_reg_model_2.predict(X_test)
eval_results(y_train, y_train_pred_ridgereg_2)
eval_results(y_test,  y_test_pred_ridgereg_2 )





















# 3. Lasso Regression
print('\n\n--------------------------\n3. Lasso Regression\n--------------------------\n')
OLS_model = sm.OLS(y_train,X_train_sm)
lasso_reg_model_1 = OLS_model.fit_regularized(L1_wt = 1.0)
print(lasso_reg_model_1.params)
print('\nstatsmodels')
y_train_pred_lassoreg_1 = lasso_reg_model_1.predict(X_train_sm)
eval_results(y_train, y_train_pred_lassoreg_1)
y_test_pred_lassoreg_1 = lasso_reg_model_1.predict(X_test_sm)
eval_results(y_test, y_test_pred_lassoreg_1)


from sklearn.linear_model import Lasso
lasso_reg_model_2 = Lasso(alpha=1, normalize=False, max_iter=1e5)
lasso_reg_model_2.fit(X,y)
print('\nscikit-learn')
y_train_pred_lassoreg_2 = lasso_reg_model_2.predict(X_train)
eval_results(y_train, y_train_pred_lassoreg_2)
y_test_pred_lassoreg_2 = lasso_reg_model_2.predict(X_test)
eval_results(y_test, y_test_pred_lassoreg_2)




# 4. Elastic Net

from sklearn.linear_model import ElasticNet
elasticnet = ElasticNet(normalize=False)
elasticnet.fit(X, y)
y_pred = elasticnet.predict(X)
rss = sum((y_pred-y)**2)
print(rss)
print([elasticnet.intercept_])
print(elasticnet.coef_)
