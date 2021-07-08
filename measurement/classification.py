import numpy as np


def compute_confusion_matrix(y, y_hat):
    y = np.reshape(y, (-1, 1))
    y_hat = np.reshape(y_hat, (-1, 1))
    classes = np.unique(y).reshape(1, -1)

    y_indicator = (y == classes) * 1
    y_hat_indicator = (y_hat == classes) * 1

    return y_hat_indicator.T @ y_indicator


def true_positives(y, y_hat, target_class):
    return (y == target_class) * (y_hat == target_class) * 1


def true_negatives(y, y_hat, target_class):
    return (y != target_class) * (y_hat != target_class) * 1


def false_positives(y, y_hat, target_class):
    return (y != target_class) * (y_hat == target_class) * 1


def false_negatives(y, y_hat, target_class):
    return (y == target_class) * (y_hat != target_class) * 1


def accuracy(y, y_hat, target_class=None):
    if target_class != None:
        tp = true_positives(y, y_hat, target_class)
        tn = true_negatives(y, y_hat, target_class)
        fp = false_positives(y, y_hat, target_class)
        fn = false_negatives(y, y_hat, target_class)
        return (tp+tn) / (tp+tn+fp+fn)
    return np.sum(y == y_hat, axis=0) / y.shape[0]


def precision(y, y_hat, target_class):
    tp = true_positives(y, y_hat, target_class)
    fp = false_positives(y, y_hat, target_class)
    return tp/(tp+fp)


def recall(y, y_hat, target_class):
    tp = true_positives(y, y_hat, target_class)
    fn = false_negatives(y, y_hat, target_class)
    return tp/(tp+fn)


def f_beta(prec, rec, beta = 1.):
    return (1. + beta*beta) / (1. / prec + beta*beta / rec)


def f1score(y, y_hat, target_class):
    p = precision(y, y_hat, target_class)
    r = recall(y, y_hat, target_class)
    return f_beta(p, r, 1)
