import numpy as np

def normal_equation_train(X, Y): # m*n m*1
    # X @ w = Y
    # X.T @ X @ w = X.T @ Y
    # w = (X.T @ X)^-1 @ X.T @ Y
    return np.linalg.inv(X.T@X)@X.T@Y
