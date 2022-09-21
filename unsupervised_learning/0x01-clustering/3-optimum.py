#!/usr/bin/env python3
"""
tests for the optimum number of clusters by variance
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Function that tests for the optimum number of clusters by variance:
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(kmin) is not int or kmin <= 0:
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if kmax is not None and (type(kmax) is not int or kmax <= 0):
        return None, None
    if type(kmax) is not int or kmax <= 0:
        return None, None
    if kmin >= kmax:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None

    results = []
    Vars = []
    for i in range(kmin, kmax + 1):
        C, clss = kmeans(X, i)
        Intra_variance = variance(X, C)
        results.append((C, clss))
        Vars.append(Intra_variance)

    d_vars = np.abs(np.array(Vars) - Vars[0])

    return results, list(d_vars)
