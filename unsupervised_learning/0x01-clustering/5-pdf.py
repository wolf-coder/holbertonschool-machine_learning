#!/usr/bin/env python3
"""
Calculates the probability density function of a Gaussian distribution.
"""
import numpy as np

def pdf(X, m, S):
    """
    function that calculates the probability density function of a Gaussian distribution:

    X is a numpy.ndarray of shape (n, d) containing the data points whose PDF should be evaluated
    m is a numpy.ndarray of shape (d,) containing the mean of the distribution
    S is a numpy.ndarray of shape (d, d) containing the covariance of the distribution
    You are not allowed to use any loops
    You are not allowed to use the function numpy.diag or the method numpy.ndarray.diagonal
    Returns: P, or None on failure
        P is a numpy.ndarray of shape (n,) containing the PDF values for each data point

    """
    
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    d = X.shape[1]
    if not isinstance(m, np.ndarray) or m.ndim != 1 or m.shape[0] != d:
        return None
    if not isinstance(S, np.ndarray) or S.ndim != 2 or S.shape != (d, d):
        return None

    X_m = X - m
    Part1 = np.linalg.inv(S) @ X_m.T
    exp = - 0.5 * np.sum(X_m * Part1.T, axis=1)
    num = np.exp(exp)
    det = np.linalg.det(S)
    pdf = num / np.sqrt(((2 * np.pi) ** d) * det)
    return  np.maximum(pdf, 1e-300) # All values in P should have a minimum value of 1e-300
