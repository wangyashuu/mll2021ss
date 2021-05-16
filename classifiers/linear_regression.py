import numpy as np

from interfaces.classifier import Classifier


def linear_regression_cost(X, y, w):
    m = y.shape[0]
    delta_y = X@w - y
    return 1/2 * 1/m * np.sum(delta_y**2, axis=0)


def linear_regression_train(X, Y, learning_rate=0.1, iteration_count=1000, batch_size=None):
    m, n = X.shape
    w = np.zeros((n, 1))

    for i in range(iteration_count): # while np.linalg.norm(w-prev_w) > 0.01
        X_chosen, Y_chosen = X, Y
        if batch_size != None:
            choices = np.random.choice(m, size=batch_size, replace=False)
            X_chosen, Y_chosen = X[choices, :], Y[choices, :]

        gradient =  X_chosen@w - Y_chosen
        w -= 1.0/(batch_size or m) * learning_rate * X_chosen.T @ gradient
        # print(linear_regression_cost(X, Y, w))
    return w


def linear_regression_predict(X, w):
    return X@w


class LinearRegressionClassifier (Classifier):

    def learn(self, X, Y, learning_rate=0.1, iteration_count=1000, batch_size=None):
        self.w = linear_regression_train(X, Y, learning_rate=0.1, iteration_count=1000, batch_size=None)

    def infer(self, X):
        return linear_regression_predict(X, self.w)
