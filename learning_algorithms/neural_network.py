import numpy as np

from interfaces.base_model import BaseModel


def ReLU(z):
    return max(0, z)

def ReLU_gradient(z):
    return 1 if z > 0 else 0

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_gradient(z):
    return sigmoid(z) * (1-sigmoid(z))

def softmax(z):
    exps = np.exp(z)
    return exps / np.sum(exps, axis=1, keepdims=True)

def softmax_gradient(z):
    return softmax(z)*(1-softmax(z))

def tanh_gradient(z):
    return 1 - np.tanh(z)**2


def neural_network_loss(Y, Y_hat):
    m = Y.shape[0]
    err = -np.sum(Y*np.log(Y_hat))
    J = 1./m * err
    return J


def neural_network_forward_propagation(X, Ws, bs, activation_functions):
    no_layers = len(Ws)
    Zs, As = [None]*no_layers, [None]*no_layers
    Zs[0] = X@Ws[0] + bs[0]
    As[0] = activation_functions[0](Zs[0])

    for i in range(1, no_layers):
        Zs[i] = As[i-1]@Ws[i] + bs[i]
        As[i] = activation_functions[i](Zs[i])

    return Zs, As


def neural_network_backward_propagation(X, Y, Ws, bs, Zs, As, activation_gradient_functions):
    no_layers = len(activation_gradient_functions)
    n, m = X.shape
    dZs, dWs, dbs = [None]*no_layers, [None]*no_layers, [None]*no_layers

    current_layer = no_layers - 1
    dZs[current_layer] = As[current_layer] - Y
    dWs[current_layer] = 1./m * As[current_layer - 1].T @ dZs[current_layer]
    dbs[current_layer] = 1./m * np.sum(dZs[current_layer], axis=0)

    for i in range(1, no_layers - 1):
        current_layer = no_layers - 1 - i
        dZs[current_layer] = dZs[current_layer + 1] @ Ws[current_layer + 1].T \
                                * activation_gradient_functions[current_layer](Zs[current_layer])
        dWs[current_layer] = 1./m * As[current_layer - 1].T @ dZs[current_layer]
        dbs[current_layer] = 1./m * np.sum(dZs[current_layer], axis=0)

    dZs[0] = dZs[1] @ Ws[1].T * activation_gradient_functions[0](Zs[0])
    dWs[0] = 1./m * X.T @ dZs[0]
    dbs[0] = 1./m * np.sum(dZs[0], axis=0)

    return dZs, dWs, dbs


def neural_network_train(X, Y, layers, batch_size=None, iteration_count=1000, learning_rate=0.1):
    m, n = X.shape
    no_layers = len(layers)
    activation_functions = [l[1] for l in layers]
    activation_gradient_functions = [l[2] for l in layers]

    # initialize the parameter
    Ws, bs = [None]*no_layers, [None]*no_layers
    no_hidden_units, _, _ = layers[0]
    Ws[0] = np.random.randn(n, no_hidden_units)
    bs[0] = np.zeros((1, no_hidden_units))
    for i in range(1, no_layers):
        no_hidden_units, _, _ = layers[i]
        no_prev_hidden_units, _, _ = layers[i-1]
        Ws[i] = np.random.randn(no_prev_hidden_units, no_hidden_units)
        bs[i] = np.zeros((1, no_hidden_units))

    # gradient descent
    for i in range(iteration_count):

        X_chosen, Y_chosen = X, Y
        if batch_size != None:
            choices = np.random.choice(m, size=batch_size, replace=False)
            X_chosen, Y_chosen = X[choices, :], Y[choices, :]

        Zs, As = neural_network_forward_propagation(X_chosen, Ws, bs, activation_functions)
        dZs, dWs, dbs = neural_network_backward_propagation(X_chosen, Y_chosen, Ws, bs, Zs, As, activation_gradient_functions)

        for i in range(len(Ws)):
            Ws[i] -= learning_rate * dWs[i]
            bs[i] -= learning_rate * dbs[i]
        # print('loss', neural_network_loss(Y_chosen, As[no_layers - 1]))
    return Ws, bs


def neural_network_predict(X, Ws, bs, layers):
    activation_functions = [l[1] for l in layers]
    _, As = neural_network_forward_propagation(X, Ws, bs, activation_functions)
    return (As[-1] > 0.5) * 1


class NeuralNetwork (BaseModel):

    def learn(self, X, Y, layers, learning_rate=0.1, iteration_count=1000, batch_size=None):
        self.layers = layers
        Ws, bs = neural_network_train(X, Y, layers, learning_rate=learning_rate, iteration_count=iteration_count, batch_size=batch_size)
        self.Ws = Ws
        self.bs = bs

    def infer(self, X):
        return neural_network_predict(X, self.Ws, self.bs, self.layers)
