import numpy as np

from interfaces.base_model import BaseModel


def support_vector_machine_eval(X, w):
    return X@w

def support_vector_machine_loss(X, y, w, big_c):
    m = y.shape[0]
    y_hat = support_vector_machine_eval(X, w)
    hinge_loss =  1./m * np.sum(((1-y)* np.maximum(0, 1 + y_hat) + y * np.maximum(0, 1 - y_hat)), axis=0)
    reg_loss = 1./2 * np.sum(w, axis=0)
    loss = big_c * hinge_loss + reg_loss
    return loss

def support_vector_machine_gradient(X, y, w, big_c):
    m = y.shape[0]
    y_hat = support_vector_machine_eval(X, w)
    hinge_loss_gradient = 1./m * (((y_hat > -1) * X).T @ (1 - y) + (-1 * (y_hat < 1) * X).T @ y).reshape(-1, 1)
    reg_loss_gradient = w
    gradient = big_c * hinge_loss_gradient + reg_loss_gradient
    return gradient

def support_vector_machine_train(X, Y, big_c=1, learning_rate=0.1, iteration_count=1000, batch_size=None):
    m, n = X.shape
    w = np.zeros((n, 1))
    Y = np.reshape(Y, (-1, 1))

    for i in range(iteration_count): # while np.linalg.norm(w-prev_w) > 0.01

        X_chosen, Y_chosen = X, Y
        if batch_size != None:
            choices = np.random.choice(m, size=batch_size, replace=False)
            X_chosen, Y_chosen = X[choices, :], Y[choices, :]

        gradient = support_vector_machine_gradient(X_chosen, Y_chosen, w, big_c)
        w -= 1.0/(batch_size or m) * learning_rate * gradient
        # print("loss:", support_vector_machine_loss(X_chosen, Y_chosen, w, big_c))
    return w


def support_vector_machine_predict(X, w):
    y_hat = support_vector_machine_eval(X, w)
    return (y_hat > 0) * 1


class SupportVectorMachine (BaseModel):

    def learn(self, X, Y, big_c=None, learning_rate=None, iteration_count=None, batch_size=None):
        self.w = support_vector_machine_train(X, Y, big_c=1, learning_rate=0.1, iteration_count=1000, batch_size=None):

    def infer(self, X):
        return support_vector_machine_predict(X, self.w)
