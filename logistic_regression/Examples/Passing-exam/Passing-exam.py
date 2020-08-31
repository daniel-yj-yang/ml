#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 21:35:32 2018

@author: Daniel Yang, Ph.D. (daniel.yj.yang@gmail.com)
"""
import argparse
from sklearn.datasets.samples_generator import make_blobs
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from statsmodels.discrete.discrete_model import Logit
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from sklearn.linear_model import LogisticRegression
import sys

print("\nLogistic Regression Assumptions:")
print("1. Dependent Variable is binary or ordinal")
print("2. Observations are independent of each other")
print("3. Little or no multicollinearity among the independent variables")
print("4. Linearity of independent variables and log odds")
# log odds = the logit = log( p/(1-p) ) = beta0 + beta1*X
# ==> p = 1 / ( 1 + exp(-[beta0+beta1*X]) )
print("5. A large sample size. It needs at minimum of 10 cases with the least frequent outcome for each independent variable in your model.")

print("\nIn contrast, it does not require the following:")
print("1. It does not need a linear relationship between the dependent and independent variables.")
print("2. The error terms (residuals) do not need to be normally distributed.")
print("3. Homoscedasticity is not required.")
print("4. The dependent variable is not measured on an interval or ratio scale.\n")


# Data Source: https://en.wikipedia.org/wiki/Logistic_regression
data = pd.read_csv(
    "/Users/daniel/My-Files/My-Projects/GitHub/ml/logistic_regression/Examples/Passing-exam/Passing-exam.csv")
y_original = data['Pass']   # Binary DV
X_original = data['Hours']  # Continuous IV

# Method 1 - Based on Scikit-Learn
X = X_original
y = y_original


def LogisticRegression_based_on_sklearn(y_pd_series, X_pd_series):
    X_data = np.reshape(X_pd_series.values, (-1, 1))  # 1D to 2D array
    y_data = y_pd_series.values
    logreg = LogisticRegression(fit_intercept=True, C=1e9)
    logreg.fit(X_data, y_data)
    print("Estimates: beta0/intercept={}, beta1/coeff={}"
          .format(logreg.intercept_, logreg.coef_))
    return (np.asscalar(logreg.intercept_), np.asscalar(logreg.coef_))


(beta0, beta1) = LogisticRegression_based_on_sklearn(y, X)


# Method 2 - Based on StatsModels
X = X_original
y = y_original


def LogisticRegression_based_on_statsmodels(y_pd_series, X_pd_series):
    # fit_intercept
    model = Logit(endog=y_pd_series, exog=add_constant(X_pd_series))
    result = model.fit()
    print(result.summary())
    print(result.summary2())
    return (result.params.values)


(beta0, beta1) = LogisticRegression_based_on_statsmodels(y, X)


# Method 4 - sklearn.linear_model.SGDClassifier

X = X_original
y = y_original

# Always scale the input. The most convenient way is to use a pipeline.
clf = make_pipeline(StandardScaler(),
                    SGDClassifier(loss="log", penalty=None, max_iter=10000, tol=1e-5))
#clf = SGDClassifier(loss="log", max_iter=1000, tol=1e-3)
clf.fit(np.array(X).reshape(-1, 1), np.array(y))
print(clf['sgdclassifier'].intercept_)
print(clf['sgdclassifier'].coef_)


# Visualization
def y_prob(x_value):
    prob = 1/(1+np.exp(-(beta0 + beta1*x_value)))
    return prob


for hours_value in range(1, 6):
    print("Studying {} hour(s) -> probability of passing the test is {}%"
          .format(hours_value, format(y_prob(hours_value)*100, '5.2f')))


x_list = list(np.arange(0, 7, .05))
y_list = list(map(lambda hours: y_prob(hours), x_list))

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.scatter(x_list, y_list, s=10, c='b', marker="s", label='fitted', alpha=0.5)
ax1.scatter(X,      y,      s=10, c='r', marker="o", label='actual', alpha=0.5)
plt.legend(loc='upper left')
plt.xlabel('hours of studying')
plt.ylabel('probability of passing')
plt.show()

sys.exit("STOP")

# Method 5 - Linear Regression

X = X_original
y = y_original


def LinearRegression_based_on_statsmodels(y_pd_series, X_pd_series):
    model = OLS(endog=y_pd_series, exog=add_constant(X_pd_series))
    result = model.fit()
    # my own residual plot
    plt.style.use('seaborn')
    plt.title('Residual Plot. Y=residual, X=predictor')
    plt.scatter(x=X_pd_series, y=result.resid)
    plt.show()
    # seaborn's residual plot
    sns.residplot(x=X_pd_series, y=y_pd_series, lowess=True, color="g")
    # summary
    print(result.summary())
    print(result.summary2())
    return (result.params.values)


(beta0, beta1) = LinearRegression_based_on_statsmodels(y, X)
