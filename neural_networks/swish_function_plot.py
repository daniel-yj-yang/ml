#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 2020

@author: daniel
"""


# swish function
# https://en.wikipedia.org/wiki/Swish_function
from numpy import arange
def swish(x, beta):
    return x/(1+exp(-beta*x))

x_sample_space = arange(-5, 5.01, 0.01)
beta = 1 # 0.5
y = swish(x_sample_space, beta)
plt.plot(x_sample_space, y)
plt.show()