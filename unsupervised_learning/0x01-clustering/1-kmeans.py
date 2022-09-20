#!/usr/bin/env python3
"""
Performs K-means on a dataset.
"""
import numpy as np
def initialize(X, k):
    """
    Function that initializes cluster centroids for K-means:
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(k) is not int or k <= 0 or k > X.shape[0]:
        return None
    n, d = X.shape
    low = np.amin(X, axis=0)
    high = np.amax(X, axis=0)
    centroids = np.random.uniform(low=low, high=high, size=(k, d))
    return centroids


def Get_clss(X, C):
    """assigne each point to a cluster """

    return np.array( [ np.argmin(np.sum(np.square(p - C), axis = 1), axis=0)  for p in X])


def kmeans(X, k, iterations=1000):
    """
Function that performs K-means on a dataset:

    X is a numpy.ndarray of shape (n, d) containing the dataset
        n is the number of data points
        d is the number of dimensions for each data point
    k is a positive integer containing the number of clusters
    iterations is a positive integer containing the maximum number of iterations that should be performed
    If no change in the cluster centroids occurs between iterations, your function should return
    Initialize the cluster centroids using a multivariate uniform distribution (based on0-initialize.py)
    If a cluster contains no data points during the update step, reinitialize its centroid
    You should use numpy.random.uniform exactly twice
    You may use at most 2 loops
    Returns: C, clss, or None, None on failure
        C is a numpy.ndarray of shape (k, d) containing the centroid means for each cluster
        
    """

    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None
    if type(k) is not int or int(k) != k or k < 1:
        return None, None
    if type(iterations) is not int\
       or int(iterations) != iterations\
           or iterations < 1:
        return None, None
    
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)

    C = initialize(X, k)
    nC = C.copy()
    for _ in range(iterations):
        clss = Get_clss(X, C)
        for i in range(k):
            indices = np.argwhere(clss == i).reshape(-1)
            if X[indices].shape[0] > 0: #  No point associated to cluster i
                nC[i] = np.mean(X[indices], axis=0)
            else:
                nC[i] = np.random.uniform(mins, maxs)
        if np.array_equal(nC, C): # No changement  
            break
        C = nC.copy()
    clss = Get_clss(X, C)
    return C, clss
    
