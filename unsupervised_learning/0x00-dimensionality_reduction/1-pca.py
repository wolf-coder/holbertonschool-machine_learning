#!/usr/bin/env python3
"""
"""

import numpy as np


def pca(X, ndim):
    """
function that performs PCA on a dataset:

    X is a numpy.ndarray of shape (n, d) where:
        n is the number of data points
        d is the number of dimensions in each point
    ndim is the new dimensionality of the transformed X
    Returns: T, a numpy.ndarray of shape (n, ndim) containing
the transformed version of X
    """
    #  Data preparation
    Data = X.copy()
    Data_m = Data - np.mean(Data, axis=0)
    #  Perform [U, S, V] = svd(Sigma)
    [U, S, V] = np.linalg.svd(Data_m)

    """
    Get T, a numpy.ndarray of shape (n, ndim) containing the
    transformed version of X
    """
    Weight_matrix = V[:ndim].T
    X_transformed = Data_m @ Weight_matrix
    return X_transformed
