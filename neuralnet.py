__author__ = 'Brian'
# My first neural net!!!

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read in data
filename = #file
data = pd.read_csv(filename)
input = data[:, :-4]
y = data[:, -4:]

# Implement feature scaling


# Initialize key values
m = input.shape[0]
j1 = input.shape[1]
j2 = 5
j3 = 4
epsilon = 0.13
numLayers = 3
a1 = []
delta1 = []
z2 = []
a2 = []
error2 = []
delta2 = []
z3 = []
a3 = []
error3 = []
targetCost = 0.0001
cost = 99999999
alpha = 0.01
regLambda = 1

# Initialize weights
theta1 = np.random.rand(j2, j1+1) * (2*epsilon) - epsilon
theta2 = np.random.rand(j3, j2+1) * (2*epsilon) - epsilon

# Sigmoid function (tanh?)
def tanh(z):
    return 2/(1 + math.exp(-z)) - 1

# Derivative of sigmoid function (dtanh?)
def dtanh(z):
    return tanh(z) * (1 - tanh(z))

# Calculate error term of hidden layer
def calcErrorTerm(thetas, errors, zs):
    return np.multiply((thetas[:, 1:].T * errors), dtanh(zs))

# Calculate the cost
def calcCost(h, y, theta1, theta2):
    m = y.shape[0]
    cost = 0
    cost += np.sum(-np.multiply(y, np.log10(h)) - np.multiply((1 - y), np.log10(1 - h))) \
            + (regLambda/(2*m)) * (np.sum(np.square(theta1)) + np.sum(np.square(theta2)))
    return cost


# Train the neural net
while (cost >= targetCost):
    h = np.empty(4)
    # Determine delta matrix for each layer
    for i in range(m):
        # Forward propagation
        a1 = [1, input[i]]
        z2 = theta1 * a1
        a2 = [1, tanh(z2)]
        z3 = theta2 * a2
        a3 = tanh(z3)
        np.vstack([h, a3])
        # Backpropagation
        actual = y[i].T
        error3 = a3 - actual
        error2 = calcErrorTerm(theta2, error3, z2)
        # Calculate adjustments for weights for this iteration
        adjustments1 = error2 * a1
        adjustments2 = error3 * a2
        # Accumulate adjustments
        delta1 += adjustments1
        delta2 += adjustments2
    # Adjust weights using regularization
    theta1[:, 0] -= alpha * (delta1 / m)
    theta1[:, 1:] -= alpha * (delta1 / m + ((regLambda/m) * theta1[:, 1:]))
    theta2[:, 0] -= alpha * (delta2 / m)
    theta2[:, 1:] -= alpha * (delta2 / m + ((regLambda/m) * theta2[:, 1:]))
    cost = calcCost(h, y, theta1, theta2)
    