#!/usr/bin/env python3
"""
Multivariate_Normal distribution parameters:
covariance, mean
"""
import numpy as np


def Cov(V0, V1, M0, M1, n):
    """
    Calculate the covariance
    """
    Coupled = [(V0[i] * V1[i]) for i in range(n)]
    return (sum(Coupled) / n) - (M0 * M1)


def mean_cov(X):
    """
    Function that calculates the mean and covariance of a data set:
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError('X must be a 2D numpy.ndarray')
    Shape = X.shape
    n, d = Shape
    if n < 2:
        raise ValueError('X must contain multiple data points')

    mean = X.mean(axis=0).reshape(1, d)
    X_mean = X - mean
    cov = (X_mean.T @ X_mean) / (n - 1)

    return mean, cov
