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
    if not isinstance(X, np.ndarray):
        raise TypeError('X must be a 2D numpy.ndarray')
    Shape = X.shape
    n, d = Shape
    if len(Shape) != 2:
        raise TypeError('X must be a 2D numpy.ndarray')
    if n < 2:
        raise ValueError('X must contain multiple data points')

    mean = [sum(X[:, i])/n for i in range(d)]

    cov = []
    for i in range(d):
        cov.append(
            [(Cov(X[:, i], X[:, j], mean[i], mean[j], n)) for j in range(d)])

    return mean, cov
