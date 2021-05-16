from scipy.sparse import csr_matrix
import numpy as np

from interfaces.classifier import Classifier


def naive_bayes_train(X, Y, ignore_min_count=0, ignore_top=0): # m*n m*1
    # TODO implement min_count
    m, n = X.shape

    classes = np.reshape(np.unique(Y), (1, -1))  # 1*c
    indexes = Y == classes # m*c

    prior_probability = np.log(indexes.sum(axis=0) / m) # done / m # 1*c

    token_counts = X.T @ indexes # done m*n @ m*c  => n*c
    sum_token_counts = token_counts.sum(axis=0)
    likelihood = np.log(token_counts + 1) - np.log(sum_token_counts + n)
    return classes, prior_probability, likelihood


def naive_bayes_predict(X, classes, prior_probability, likelihood):
    score = prior_probability + X @ likelihood
    index = np.asarray(np.argmax(score, axis=1).T)
    return classes[:, np.squeeze(index)].T, index


class NaiveBayesClassifier (Classifier):

    def learn(self, X, Y, ignore_min_count=0, ignore_top=0):
        classes, prior_probability, likelihood = naive_bayes_train(X, Y)
        self.classes = classes
        self.prior_probability = prior_probability
        self.likelihood = likelihood

    def infer(self, X):
        return naive_bayes_predict(X, self.classes, self.prior_probability, self.likelihoods)
