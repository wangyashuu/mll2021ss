import numpy as np

from interfaces.base_model import BaseModel

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def logistic_regression_eval(X, w)
    return sigmoid(X@w)


def logistic_regression_loss(y, y_hat):
    m = y.shape[0]
    loss = - (y*np.log(y_hat) + (1-y)*np.log(1-y_hat))
    return 1.0/m * np.sum(loss, axis=0)


def logistic_regression_train(X, Y, learning_rate=0.1, iteration_count=1000, batch_size=None):
    m, n = X.shape
    w = np.zeros((n, 1))

    for i in range(iteration_count):

        X_chosen, Y_chosen = X, Y
        if batch_size != None:
            choices = np.random.choice(m, size=batch_size, replace=False)
            X_chosen, Y_chosen = X[choices, :], Y[choices, :]

        Y_hat = logistic_regression_eval(X_chosen, w)
        gradient = X_chosen.T @ (Y_hat - Y_chosen)
        w -= 1.0/(batch_size or m) * learning_rate * gradient
        # print(logistic_regression_loss(Y_chosen, Y_hat))
    return w


def logistic_regression_predict(X, w):
    return (logistic_regression_eval(X, w) > 0.5) * 1


class LogisticRegression (BaseModel):

    def learn(self, X, Y, learning_rate=0.1, iteration_count=1000, batch_size=None):
        self.w = logistic_regression_train(X, Y, learning_rate=learning_rate, iteration_count=iteration_count, batch_size=batch_size)

    def infer(self, X):
        return logistic_regression_predict(X, self.w)
