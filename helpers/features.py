import numpy as np

def extract_raw_features_and_targets(files, classname = ""):
    X, Y = [], []
    for name in files:
        if isinstance(files[name], str):
            X.append(files[name])
            Y.append(classname)
        else:
            X_sub, Y_sub = extract_raw_features_and_targets(files[name], classname + name)
            X = X + X_sub
            Y = Y + Y_sub
    return X, Y


def mean_normalizer(of_X):
    mean = np.mean(of_X, axis=0)
    scale_range = np.max(of_X, axis=0) - np.min(of_X, axis=0)
    return lambda X: (X-mean) / scale_range


def z_score_normalizer(of_X): # m*n
    mean = np.mean(of_X, axis=0)
    scale_range = np.std(of_X, axis=0)
    return lambda X: (X-mean) / scale_range
