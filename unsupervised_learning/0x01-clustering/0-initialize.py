#!/usr/bin/env python3
"""
Initializing cluster centroids for K-means
"""
import numpy as np


def initialize(X, k):
    """
    Function that initializes cluster centroids for K-means:
    """
    # if not isinstance(X, np.ndarray) or X.ndim != 2:
    #     return None
    # if not isinstance(k, int) or k < 1 or k > X.shape[0]:
    #     return None
    # d = X.ndim
    # Low = np.amin(X, axis=0)
    # High = np.amax(X, axis=0)
    # centroides = np.random.uniform(Low, High, size=(k, d))
    # return centroides
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(k) is not int or k <= 0 or k > X.shape[0]:
        return None
    n, d = X.shape
    low = np.amin(X, axis=0)
    high = np.amax(X, axis=0)
    centroids = np.random.uniform(low=low, high=high, size=(k, d))
    return centroids
