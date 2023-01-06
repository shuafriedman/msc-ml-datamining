#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 09:14:13 2019

@author: yosi
"""

import numpy as np
import pandas as pd
# import seaborn as sns
# import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

table = pd.read_csv('/home/shua/Desktop/msc-ml-datamining/Machine_Learning/Week4-LogisticRegression/Intel-ML101-Class4/data/weight-height.csv',skiprows =0)
height = table['Height'][table['Gender'] == 'Male'].to_numpy()
weight = table['Weight'][table['Gender'] == 'Male'].to_numpy()

X = (weight-np.mean(weight))/np.std(weight) #+0.5 #play with bias 
Y = (height-np.mean(height))/np.std(height)

# Explicit minimization

phase_1 = np.matmul(X.reshape(-1,1),X.reshape(1,-1))
# Use pseudo-nverse to avoid singularities
phase_2 = np.linalg.pinv(phase_1)
phase_3 = np.matmul(phase_2,X.reshape(-1,1))
W_opt = np.matmul(Y.reshape(1,-1),phase_3)


# Initialization
W = 0
b = 0
L = 0.015  # The learning Rate
epochs = 100  # The number of iterations to perform gradient descent

n = float(len(X)) # Number of elements in X

b_ = []
W_ = []
# Performing Gradient Descent
for i in range(epochs):
    Y_pred = W*X + b  # The current predicted value of Y
    D_W = (-2/n) * sum(X * (Y - Y_pred))  # Derivative wrt W
    D_b = (-2/n) * sum(Y - Y_pred)  # Derivative wrt b
    W = W - L * D_W  # Update W
    b = b - L * D_b  # Update b
    W_.append(W)
    b_.append(b)

plt.plot(W_)
plt.plot(b_)

plt.plot(X,Y,'.')
plt.plot(X, X*W+b,'r')

# %matplotlib qt5

fig, ax = plt.subplots(figsize=(5, 3))
x_values = np.linspace( -4, 4, 100)
ax.set(xlim= (-4, 4), ylim= (-4, 4))
plt.plot(X,Y,'.')
plt.plot(x_values, x_values*W_opt[0], 'y--', linewidth=4)
    
line, = ax.plot(x_values, x_values*W_[i]+b_[i], 'r-', linewidth=2)

def animate(i):
    label = 'epoch {0}'.format(i)
    W_i = W_[i]
    b_i = b_[i]
    line.set_ydata(x_values*W_[i]+b_[i])  # update the data.
    ax.set_xlabel(label)
    return line, ax

anim = animation.FuncAnimation(
    fig, animate, frames=range(epochs-1), interval = 100, repeat=False) 

import os
print(os.getcwd())
anim.save(os.getcwd() + '/lms.gif', writer = 'imagemagick')
