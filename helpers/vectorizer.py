import numpy as np
import collections
from scipy.sparse import csr_matrix

from .text import tokenize

def vectorize_words(token_counts, vocabulary):
    result = np.zeros((1, len(vocabulary)), dtype=int)
    for token in token_counts:
        if token in vocabulary:
            result[:, vocabulary[token]] = token_counts[token]
    return result


def vectorize_texts_by_vocabulary(text_array, vocabulary):
    data, indices = [], []
    indptr = [0]
    for text in text_array:
        token_counts = collections.Counter(tokenize(text))
        for token in token_counts:
            if token in vocabulary:
                indices.append(vocabulary[token])
                data.append(token_counts[token])
        indptr.append(len(data))
    matrix = csr_matrix((data, indices, indptr), shape=(len(text_array), len(vocabulary)))
    return matrix


def vectorize_text_by_vocabulary(text, vocabulary):
    tokens = tokenize(text)
    freq = collections.Counter(tokens)
    return vectorize_words(freq, vocabulary)


def vectorize_texts(text_array):
    vocabulary = {}
    data, indices = [], []
    indptr = [0]
    for text in text_array:
        token_counts = collections.Counter(tokenize(text))
        for token in token_counts:
            index = vocabulary.setdefault(token, len(vocabulary))
            indices.append(index)
            data.append(token_counts[token])
        indptr.append(len(data))
    matrix = csr_matrix((data, indices, indptr), shape=(len(text_array), len(vocabulary)))
    return matrix, vocabulary


def vectorize_by_grouping(arr):
    bags, result = np.unique(arr, return_inverse=True)
    return np.reshape(result, arr.shape), bags
