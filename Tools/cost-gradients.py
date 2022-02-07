import numpy as np
import pandas as pd


#Training 1
X = pd.DataFrame({'temp': [1000, 100], 'den': [1e5, 1e4]}) #X = feature matrix
M =  #Number of training instances
phi = #parameters
y = #target vector or predicted value
alpha = #step direction
C = #cost function

def grad_of_cost_func(X):
    C = (2 / M) * M.transpose * (X*phi - y)
    return C

def batch_grad_descent():
    phi(t-1) - alpha * (C / phi)

