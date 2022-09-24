#!/usr/bin/env python3
"""
Calculating the expectation step in the EM algorithm for a GMM.
"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
Function that calculates the expectation step in the EM algorithm for a GMM
    """

    n = X.shape[0] # Number of data points
    k = pi.shape[0] #  Pi contains the priors for each cluster, initialized evenly
    Posteriors = np.zeros((k, n))  # Initialization

    for i in range(k):
        Posteriors[i] = pi[i] * pdf(X, m[i], S[i])
    Log = np.sum(Posteriors, axis=0, keepdims=True) # l is the total log likelihood
    Posteriors = Posteriors / Log
    Log = np.sum(np.log(Log))
    return Posteriors, Log
