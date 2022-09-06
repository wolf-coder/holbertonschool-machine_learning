#!/usr/bin/env python3
"""
Representing a Multivariate Normal distribution.
"""
import numpy as np


class MultiNormal():
    """
    class that represents a Multivariate Normal distribution:
    """
    def __init__(self, data):
        """
    class constructor def __init__(self, data):
        data is a numpy.ndarray of shape (d, n) containing the data set:
        n is the number of data points
        d is the number of dimensions in each data point
        If data is not a 2D numpy.ndarray, raise a TypeError with the message
data must be a 2D numpy.ndarray
        If n is less than 2, raise a ValueError with the message
data must contain multiple data points
    Set the public instance variables:
        mean - a numpy.ndarray of shape (d, 1) containing the mean of data
        cov - a numpy.ndarray of shape (d, d) containing the covariance
matrix data
    You are not allowed to use the function numpy.cov

        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError('data must be a 2D numpy.ndarray')
        d, n = data.shape
        if n < 2:
            raise ValueError('data must contain multiple data points')

        self.mean = data.mean(axis=1, keepdims=True).reshape(d, 1)
        Data_mean = data - self.mean
        self.cov = (Data_mean @ Data_mean.T) / (n - 1)

    def pdf(self, x):
        """
calculates the PDF at a data point:
    x is a numpy.ndarray of shape (d, 1) containing the data point
whose PDF should be calculated
        d is the number of dimensions of the MultiNormal instance
    If x is not a numpy.ndarray, raise a TypeError with the
message x must be a numpy.ndarray
    If x is not of shape (d, 1), raise a ValueError with the
message x must have the shape ({d}, 1)
    Returns the value of the PDF
    You are not allowed to use the function numpy.cov

        """
        if not isinstance(x, np.ndarray):
            raise TypeError('x must be a numpy.ndarray')
        dim, _ = self.cov.shape

        if len(x.shape) != 2 or x.shape[1] != 1 or x.shape[0] != dim:
            raise ValueError("x must have the shape ({}, 1)".format(dim))

        p2 = ((x - self.mean).T @ np.linalg.inv(self.cov) @ (x - self.mean))
        p1 = np.exp((-1 / 2) * p2)
        p3 = (np.sqrt(np.linalg.det(self.cov)))
        pdf = 1 / (((2 * np.pi) ** (dim / 2)) * p3) * p1

        return pdf[0][0]
