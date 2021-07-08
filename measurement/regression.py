import numpy as np


def r2_score(y, y_hat):
    m = y.shape[0]
    y_bar = np.sum(y) / m
    r2 = 1 - (np.sum((y - y_hat)**2) / np.sum((y - y_bar)**2))
    return r2
