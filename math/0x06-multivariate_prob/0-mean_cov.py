#!/usr/bin/env python3
"""
"""


def Cov(V0, V1, M0, M1, n):
    """
    Calculate the covariance
    """
    Coupled = [(V0[i] * V1[i]) for i in range(n)]
    return (sum(Coupled) / n) - (M0 * M1)


def mean_cov(X):
    """
    * Function that calculates the mean and covariance of a data set:
        * X is a numpy.ndarray of shape (n, d) containing the data set:
            * n is the number of data points
            * d is the number of dimensions in each data point
            * If X is not a 2D numpy.ndarray,
raise a TypeError with the message X must be a 2D numpy.ndarray
            * If n is less than 2, raise a ValueError with the message
X must contain multiple data points
        * Returns: mean, cov:
            * mean is a numpy.ndarray of shape (1, d)
containing the mean of the data set
            * cov is a numpy.ndarray of shape (d, d)
containing the covariance matrix of the data set
        * You are not allowed to use the function numpy.cov.
    """

    Shape = X.shape
    n, d = Shape
    if len(Shape) != 2:
        raise TypeError('the message X must be a 2D numpy.ndarray')
    if n < 2:
        raise ValueError('the message X must contain multiple data points')

    mean = [sum(X[:, i])/n for i in range(d)]

    cov = []
    for i in range(d):
        cov.append(
            [(Cov(X[:, i], X[:, j], mean[i], mean[j], n)) for j in range(d)])

    return mean, cov
