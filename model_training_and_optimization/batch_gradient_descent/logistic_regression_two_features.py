#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 21:35:32 2018

@author: Daniel Yang, Ph.D. (daniel.yj.yang@gmail.com)
"""

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score


iris = sklearn.datasets.load_iris()
X = iris.data[:, :2]
y = (iris.target != 0) * 1

# https://medium.com/@martinpella/logistic-regression-from-scratch-in-python-124c5636b8ac

# Method 1 - batch gradient descent


class Logistic_regression_as_optimized_by_batch_gradient_descent:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False, plot_decision_boundary_during_training=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.plot_decision_boundary_during_training = plot_decision_boundary_during_training

    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def fit(self, X, y):
        self.X = X
        self.y = y

        if self.fit_intercept:
            self.X = self.__add_intercept(self.X)

        # weights initialization
        self.theta = np.zeros(self.X.shape[1])
        self.lossHistory = []

        for i in range(self.num_iter):
            z = np.dot(self.X, self.theta)
            h = self.__sigmoid(z)  # y-hat
            error = h - self.y
            loss = np.sum(error ** 2)
            self.lossHistory.append(loss)

            gradient = np.dot(self.X.T, error) / self.y.size
            self.theta -= self.lr * gradient

            if(self.plot_decision_boundary_during_training):
                self.plot_decision_boundary(epoch=i)

            if(self.verbose == True and i % 10000 == 0):
                z = np.dot(self.X, self.theta)
                h = self.__sigmoid(z)
                print(f'loss: {self.__loss(h, self.y)} \t')

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        return self.__sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold=0.5):
        return self.predict_prob(X) >= threshold

    def plot_decision_boundary(self, epoch=None):

        # references:
        # https://www.pyimagesearch.com/2016/10/10/gradient-descent-with-python/
        # https://towardsdatascience.com/building-a-logistic-regression-in-python-301d27367c24

        # plot the original data
        fig = plt.figure()
        plt.scatter(self.X[:, 1], self.X[:, 2], marker="o", c=self.y)

        # the decision boundary
        # y_hat = h(x) = 1 / (1 + exp(-z))
        # if z > 0, h(x) > 0.5
        # if z < 0, h(x) < 0.5
        # also, z = theta.T * X

        # z = 0 --> the decision boundary
        # when there are only two features, z = theta0 + theta1 x1 + theta2 x2
        # solve for z = 0 --> x2 = (- theta0 - theta1 x1) / theta2

        X1_values = [np.min(self.X[:, 1]) * 0.99, np.max(self.X[:, 1]) * 1.01]
        X2_values = (-self.theta[0] -
                     np.dot(self.theta[1], X1_values)) / self.theta[2]
        plt.plot(X1_values, X2_values, color='red',
                 linestyle='dashed', label='Decision Boundary')
        if(epoch != None):
            fig.suptitle('Epoch #{}'.format(epoch))
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()
        plt.show()

    def plot_loss_history(self):
        # construct a figure that plots the loss over time
        fig = plt.figure()
        plt.plot(np.arange(0, self.num_iter), self.lossHistory)
        fig.suptitle("Training Loss")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.show()


# 1. Batch gradient descent
# https://medium.com/@martinpella/logistic-regression-from-scratch-in-python-124c5636b8ac
model = Logistic_regression_as_optimized_by_batch_gradient_descent(
    lr=0.01, num_iter=50, plot_decision_boundary_during_training=True)
model.fit(X, y)
model.plot_decision_boundary()
model.plot_loss_history()

print(model.theta)
predicted_classes = model.predict(X)
accuracy = accuracy_score(y.flatten(), predicted_classes)
print(accuracy)


# 2. Sklearn
model = sklearn.linear_model.LogisticRegression(
    fit_intercept=True, C=1e20, penalty='none')
model.fit(X, y)
print(model.intercept_, model.coef_)
predicted_classes = model.predict(X)
accuracy = accuracy_score(y.flatten(), predicted_classes)
print(accuracy)


# next1: penalty
# next2: multi-class


# 3. Another batch gradient descent
# https://towardsdatascience.com/building-a-logistic-regression-in-python-301d27367c24
# https://www.pyimagesearch.com/2016/10/10/gradient-descent-with-python/

# 4. Mini-batch gradient descent
# https://www.pyimagesearch.com/2016/10/17/stochastic-gradient-descent-sgd-with-python/
