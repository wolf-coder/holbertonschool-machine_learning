#!/usr/bin/env python3
"""
Representing a Multivariate Normal distribution.
"""
import numpy as np


class MultiNormal():
    """
class that represents a Multivariate Normal distribution:
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
    def __init__(self, data):
        "constructor method"
        if not isinstance(data, np.ndarray) or data.ndim != 2\
           or data.shape[0] < 2:
            raise TypeError('data must be a 2D numpy.ndarray')
        d, n = data.shape
        if n < 2:
            raise ValueError('data must contain multiple data points')

        self.mean = data.mean(axis=1).reshape(3, 1)
        Data_mean = data - self.mean
        self.cov = (Data_mean @ Data_mean.T) / (n - 1)
