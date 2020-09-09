#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 10:01:51 2020

@author: Daniel Yang Ph.D (daniel.yj.yang@gmail.com)
"""

# generate variables
from numpy import mean
from numpy import std
from numpy.random import randn
from numpy.random import seed
from matplotlib import pyplot as plt
# seed random number generator
seed(1230293)
# prepare data
n = 10000
var1 = 20 * randn(n) + 50
var2 = var1 + (10 * randn(n) + 20)
# summarize
print('variable1: mean=%.3f stdv=%.3f' % (mean(var1), std(var1)))
print('variable2: mean=%.3f stdv=%.3f' % (mean(var2), std(var2)))
# plot
#create a new figure
plt.figure(figsize=(7,7))
plt.scatter(var1, var2)
plt.show()

# PCA
import pandas as pd
from numpy.linalg import svd
from numpy import transpose

X_raw = pd.DataFrame({'var1': var1, 'var2': var2})
X_colMeans = mean(X_raw, axis = 0)
#X_colStds = std(X_raw, axis=0)
X = (X_raw - X_colMeans) #/ X_colStds
U, D, V_t = svd(X, full_matrices=False)
W = transpose(V_t) # eigenvectors
L = D**2 / (n-1) # eigenvalues

#create a new figure
plt.figure(figsize=(7,7))
plt.scatter(X_raw['var1'], X_raw['var2'])
plt.axis('equal')

slope_PC1 = W[1,0]/W[0,0] # a vector pointing toward (W[0,0], W[1,0])
intercept_PC1 = X_colMeans[1] - X_colMeans[0]* slope_PC1

plt.plot([ min(X_raw['var1']),                           max(X_raw['var1'])                           ],
         [ intercept_PC1 + slope_PC1*min(X_raw['var1']), intercept_PC1 + slope_PC1*max(X_raw['var1']) ],
         'r')

slope_PC2 = W[1,1]/W[0,1] # a vector pointing toward (W[0,1], W[1,1])
intercept_PC2 = X_colMeans[1] - X_colMeans[0]* slope_PC2

plt.plot([ min(X_raw['var1']),                           max(X_raw['var1'])                           ],
         [ intercept_PC2 + slope_PC2*min(X_raw['var1']), intercept_PC2 + slope_PC2*max(X_raw['var1']) ],
         'r')

plt.show()

# PCA
from sklearn.decomposition import PCA
import numpy as np
pca = PCA(n_components=2)
PCs = pca.fit_transform(X)
pca.components_ # eigenvectors
pca.explained_variance_ # eigenvalues
#create a new figure
plt.figure(figsize=(7,7))
plt.scatter(PCs[:,0], PCs[:,1])
plt.axis('equal')
plt.plot([ np.min(PCs), np.max(PCs) ],
         [ 0,             0   ],
         'r')
plt.plot([ 0,             0   ],
         [ np.min(PCs), np.max(PCs) ],
         'r')
plt.show()



# Big Mart Sales
import pandas as pd
X_raw = pd.read_csv('/Users/daniel/Data-Science/Data/Retail/Big_Mart_Sales/Python/train.csv')
for col in X_raw.columns:
    print(col)
X_raw = X_raw.drop(columns=['Item_Identifier','Item_Outlet_Sales'])
X_colMeans = mean(X_raw, axis = 0)
X_colStds = std(X_raw, axis=0)
X = (X_raw - X_colMeans) / X_colStds
pca = PCA(n_components=2)
PCs = pca.fit_transform(X)
#create a new figure
plt.figure(figsize=(7,7))
plt.scatter(PCs[:,0], PCs[:,1])
plt.axis('equal')
plt.show()



# https://stackoverflow.com/questions/38698277/plot-normal-distribution-in-3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
#from mpl_toolkits.mplot3d import Axes3D
from numpy import sqrt

#Parameters to set
mu_x = 0
variance_x = 3

mu_y = 0
variance_y = 15

r_xy = 0
covariance_xy = sqrt(variance_x) * sqrt(variance_y) * r_xy

#Create grid and multivariate normal
x = np.linspace(-10,10,500)
y = np.linspace(-10,10,500)
X, Y = np.meshgrid(x,y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X; pos[:, :, 1] = Y
rv = multivariate_normal([mu_x, mu_y], [[variance_x, covariance_xy], [covariance_xy, variance_y]])
Z = rv.pdf(pos)

#Make a 3D plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()

#PCA
X_raw = pd.DataFrame({ 'X': X.flatten(), 'Y': Y.flatten(), 'Z': Z.flatten() })
X_colMeans = mean(X_raw, axis = 0)
X_colStds = std(X_raw, axis=0)
X = (X_raw - X_colMeans) / X_colStds
pca = PCA(n_components=2)
PCs = pca.fit_transform(X)
#create a new figure
plt.figure(figsize=(7,7))
plt.scatter(PCs[:,0], PCs[:,1])
plt.axis('equal')
plt.show()