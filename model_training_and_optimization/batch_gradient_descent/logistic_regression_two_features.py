#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 21:35:32 2018

@author: Daniel Yang, Ph.D. (daniel.yj.yang@gmail.com)
"""

import matplotlib.pyplot as plt
from matplotlib import animation, rc
import numpy as np
import sklearn
from sklearn import datasets
from sklearn.metrics import accuracy_score

plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
plt.rcParams['animation.convert_path'] = '/usr/local/bin/convert'
plt.rcParams.update({'font.size': 20})

# equivalent to rcParams['animation.html'] = 'html5'
rc('animation', html='html5')

iris = datasets.load_iris()
X = iris.data[:, :2]
y = (iris.target != 0) * 1

# modified but a very early version was based on https://medium.com/@martinpella/logistic-regression-from-scratch-in-python-124c5636b8ac

# Method 1 - batch gradient descent


class Logistic_regression_as_optimized_by_batch_gradient_descent:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose

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
        self.training_History = []

        for i in range(self.num_iter):
            z = np.dot(self.X, self.theta)
            h = self.__sigmoid(z)  # y-hat
            error = h - self.y
            loss = np.sum(error ** 2)

            #self.training_History.append([loss, self.theta.tolist()])

            gradient = np.dot(self.X.T, error) / self.y.size
            self.theta -= self.lr * gradient

            # self.theta = [0,0,0] is not good for plotting, so starting from here
            self.training_History.append([loss, self.theta.tolist()])

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
        plt.figure(figsize=(8, 8))
        plt.scatter(self.X[:, 1], self.X[:, 2], marker="o", c=self.y)

        # the decision boundary
        # y_hat = h(x) = 1 / (1 + exp(-z))
        # if z > 0, h(x) > 0.5
        # if z < 0, h(x) < 0.5
        # also, z = theta.T * X

        # z = 0 --> the decision boundary
        # when there are only two features, z = theta0 + theta1 x1 + theta2 x2
        # solve for z = 0 --> x2 = (- theta0 - theta1 x1) / theta2

        X1_values = [np.min(self.X[:, 1])-0.1, np.max(self.X[:, 1])+0.1]
        X2_values = (-self.theta[0] -
                     np.dot(self.theta[1], X1_values)) / self.theta[2]
        plt.plot(X1_values, X2_values, color='red',
                 linestyle='dashed', label='Decision Boundary')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()
        plt.show()

    def plot_loss_history(self):
        # construct a figure that plots the loss over time
        plt.figure(figsize=(8, 8))
        plt.plot(np.arange(0, len(self.training_History)), np.array(
            self.training_History)[:, 0], label='Training Loss')
        plt.legend(loc=1)
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.show()

    # see also https://xavierbourretsicotte.github.io/animation_ridge.html
    def animate_decision_boundary(self):
        # First set up the figure, the axis, and the plot element we want to animate
        fig1, ax1 = plt.subplots(figsize=(8, 8))
        line, = ax1.plot([], [], color='red', linestyle='dashed',
                         label='Decision Boundary', lw=1.5)
        point, = ax1.plot([], [], '*', color='red', markersize=4)
        value_display = ax1.text(0.02, 0.02, '', transform=ax1.transAxes)
        # https://stackoverflow.com/questions/9401658/how-to-animate-a-scatter-plot?rq=1
        scatterplot = ax1.scatter([], [], marker="o")
        plt.xlabel('X1')
        plt.ylabel('X2')

        def init_1():
            line.set_data([], [])
            point.set_data([], [])
            value_display.set_text('')
            scatterplot.set_offsets([])

            return line, point, value_display, scatterplot

        def update_1(epoch):
            # Animate line
            X1_values = [np.min(self.X[:, 1])-0.1, np.max(self.X[:, 1])+0.1]
            this_theta = self.training_History[epoch][1]
            # print(this_theta)
            X2_values = (-this_theta[0] -
                         np.dot(this_theta[1], X1_values)) / this_theta[2]
            line.set_data(X1_values, X2_values)
            #print(epoch, X1_values, X2_values)

            # Animate points
            point.set_data([], [])

            # Animate value display
            value_display.set_text('Epoch # {}'.format(epoch+1))

            # scatterplot
            scatterplot.set_offsets(self.X[:, 1:3])
            scatterplot.set_array(self.y)

            # https://stackoverflow.com/questions/43674917/animation-in-matplotlib-with-scatter-and-using-set-offsets-autoscale-of-figure
            # scale
            ax1.set_xlim(min(min(self.X[:, 1]), min(
                X1_values))-0.1, max(max(self.X[:, 1]), max(X1_values))+0.1)
            ax1.set_ylim(min(min(self.X[:, 2]), min(
                X2_values))-0.1, max(max(self.X[:, 2]), max(X2_values))+0.1)

            return line, point, value_display, scatterplot

        ax1.legend(loc="upper left")

        # blit=True means only re-draw the parts that have changed.
        anim1 = animation.FuncAnimation(fig1, update_1, init_func=init_1,
                                        frames=len(self.training_History),
                                        interval=20,
                                        blit=True)

        anim1.save('/Users/daniel/tmp/animation.gif',
                   fps=60, writer='imagemagick')

        return


# 1. Batch gradient descent
# https://medium.com/@martinpella/logistic-regression-from-scratch-in-python-124c5636b8ac
model = Logistic_regression_as_optimized_by_batch_gradient_descent(
    lr=0.1, num_iter=500)
model.fit(X, y)
model.animate_decision_boundary()

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
