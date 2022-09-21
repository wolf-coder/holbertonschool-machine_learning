#!/usr/bin/env python3
"""
Performs K-means on a dataset.
"""
import numpy as np

def variance(X, C):
    """
Function that calculates the total intra-cluster variance for a data set:

    X is a numpy.ndarray of shape (n, d) containing the data set
    C is a numpy.ndarray of shape (k, d) containing the centroid means for each cluster
    You are not allowed to use any loops
    Returns: var, or None on failure
        var is the total variance
    """
    if type(X) is not np.ndarray or X.ndim != 2:
        return None
    if type(C) is not np.ndarray or C.ndim != 2 or C.shape[1] != X.shape[1]:
        return None
    Squared_distance = np.square(X[:,np.newaxis] - C)
    Sum_Squared_distance = np.sum(Squared_distance, axis=2)
    Least_squared = np.min(Sum_Squared_distance, axis=1)   
    var = np.sum(Least_squared)
    return var
