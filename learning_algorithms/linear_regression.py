import numpy as np

from interfaces.base_model import BaseModel

def linear_regression_eval(X, w):
    return X@w

def linear_regression_cost(y, y_hat):
    m = y.shape[0]
    delta_y = y_hat - y
    return 1/2 * 1/m * np.sum(delta_y**2, axis=0)


def linear_regression_train(X, Y, learning_rate=0.1, iteration_count=1000, batch_size=None):
    m, n = X.shape
    w = np.zeros((n, 1))

    for i in range(iteration_count): # while np.linalg.norm(w-prev_w) > 0.01

        X_chosen, Y_chosen = X, Y
        if batch_size != None:
            choices = np.random.choice(m, size=batch_size, replace=False)
            X_chosen, Y_chosen = X[choices, :], Y[choices, :]

        Y_hat = linear_regression_eval(X_chosen, w)
        gradient =  X_chosen.T @ (Y_hat - Y_chosen)
        w -= 1.0/(batch_size or m) * learning_rate * gradient
        # print(linear_regression_cost(Y, Y_hat))
    return w


def linear_regression_predict(X, w):
    return linear_regression_eval(X, w)


class LinearRegression (BaseModel):

    def learn(self, X, Y, learning_rate=0.1, iteration_count=1000, batch_size=None):
        self.w = linear_regression_train(X, Y, learning_rate=learning_rate, iteration_count=iteration_count, batch_size=batch_size)

    def infer(self, X):
        return linear_regression_predict(X, self.w)
