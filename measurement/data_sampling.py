import numpy as np


def get_k_folds(n, k, random=True):
    fold_size = n//k
    m = fold_size*k

    indices = np.random.permutation(m) if random else np.arange(m)
    indices_splits = indices.reshape(k, -1)

    folds = []
    fold_indices = np.arange(k)
    for i in range(k):
        train_indices = indices_splits[fold_indices != i].flatten()
        test_indices = indices_splits[fold_indices == i].flatten()
        folds.append((train_indices, test_indices))

    return folds
