#!/usr/bin/env python3
"""
Calculating the expectation step in the EM algorithm for a GMM.
"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
Function that calculates the expectation step in the EM algorithm for a GMM:

    X is a numpy.ndarray of shape (n, d) containing the data set
    pi is a numpy.ndarray of shape (k,) containing the priors for each cluster
    m is a numpy.ndarray of shape (k, d) containing the centroid
means for each cluster
    S is a numpy.ndarray of shape (k, d, d) containing the covariance
matrices for each cluster
    You may use at most 1 loop
    Returns: g, l, or None, None on failure
        g is a numpy.ndarray of shape (k, n) containing the
posterior probabilities
for each data point in each cluster
        l is the total log likelihood
    """
    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None
    if type(pi) is not np.ndarray or len(pi.shape) != 1:
        return None, None
    if type(m) is not np.ndarray or m.ndim != 2:
        return None, None
    if type(S) is not np.ndarray or S.ndim != 3:
        return None, None
    if not np.isclose([np.sum(pi)], [1])[0]:
        return None, None

    n = X.shape[0]  # Number of data points
    k = pi.shape[0]  # Pi contains the priors for each cluster
    Posteriors = np.zeros((k, n))  # Initialization

    for i in range(k):
        Posteriors[i] = pi[i] * pdf(X, m[i], S[i])

    # Posteriors is the total log likelihood
    Log = np.sum(Posteriors, axis=0, keepdims=True)

    Posteriors = Posteriors / Log
    Log = np.sum(np.log(Log))
    return Posteriors, Log
