#!/usr/bin/env python3
"""
Initializing cluster centroids for K-means
"""
import numpy as np


def initialize(X, k):
    """
Function that initializes cluster centroids for K-means:

    X is a numpy.ndarray of shape (n, d) containing the dataset that
will be used for K-means clustering
        n is the number of data points
        d is the number of dimensions for each data point
    k is a positive integer containing the number of clusters
    The cluster centroids should be initialized with a multivariate
niform distribution along each dimension in d:
        The minimum values for the distribution should be the minimum
alues of X along each dimension in d
        The maximum values for the distribution should be the maximum
alues of X along each dimension in d
        You should use numpy.random.uniform exactly once
    You are not allowed to use any loops
    Returns: a numpy.ndarray of shape (k, d) containing
the initialized centroids for each cluster, or None on failure
    """
    if type(X) is not np.ndarray or X.ndim != 2:
        return None
    if not isinstance(k, int) or k < 1:
        return None
    d = X.ndim
    Low = np.amin(X, axis=0)
    High = np.amax(X, axis=0)
    centroides = np.random.uniform(Low, High, size=(k, d))
    return centroides
