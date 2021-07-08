from scipy.sparse import csr_matrix
import numpy as np

from interfaces.base_model import BaseModel


def naive_bayes_train(X, Y, ignore_min_count=0, ignore_top=0): # m*n m*1
    """
    args:
        X: (m, n) matrix
        Y: (m, 1) matrix
    returns:
        classes: an array of classes of Y. (1, c) matrix where c is number of classes
        prior_probability: probability of classes, denote as P(C).
        likelihood: log likelihood of each feature (given each class), denotes as  P(w|C)
    """
    # TODO implement min_count
    m, n = X.shape
    classes = np.reshape(np.unique(Y), (1, -1))

    # one-hot encoding of Y
    # (m, c) matrix, class_indicator[i][j] indicates if example i is clcass j
    class_indicator = Y == classes

    # P(C) probability of each class
    # (1, c) matrix, prior_probability[j] indicates the probability of class j
    prior_probability = np.log(indexes.sum(axis=0) / m)

    # sum up the counts of each feature (in each class).
    # (n, c) matrix, feature_counts[i][j] indicates counts of feature i in class j
    feature_counts = X.T @ class_indicator

    # sum up of counts of all feature (in each class)
    # (1, c) matrix, sum_feature_counts[j] indicates counts of all features in class j
    sum_feature_counts = feature_counts.sum(axis=0)

    # P(w|C) log likelihood of each feature (given each class)
    # (n, c) matrix, likelihood[i][j] indicates log likelihood of feature i given class j
    likelihood = np.log(feature_counts + 1) - np.log(sum_feature_counts + n)
    return classes, prior_probability, likelihood


def naive_bayes_predict(X, classes, prior_probability, likelihood):
    # compute P(C)*P(w|C) for X with given features w
    score = prior_probability + X @ likelihood
    index = np.asarray(np.argmax(score, axis=1).T)
    return classes[:, np.squeeze(index)].T, index


class NaiveBayes (BaseModel):

    def learn(self, X, Y, ignore_min_count=0, ignore_top=0):
        classes, prior_probability, likelihood = naive_bayes_train(X, Y)
        self.classes = classes
        self.prior_probability = prior_probability
        self.likelihood = likelihood

    def infer(self, X):
        return naive_bayes_predict(X, self.classes, self.prior_probability, self.likelihoods)
