#!/usr/bin/env python3
"""
Calculating the expectation step in the EM algorithm for a GMM.
"""

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
Function that calculates the expectation step in the EM algorithm for a GMM:Function that calculates the expectation step in the EM algorithm for a GMM:

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
    n, d = X.shape

    if pi.shape[0] > n:
        return None, None

    k = pi.shape[0]



    if not np.isclose([np.sum(pi)], [1])[0]:
        return None, None

    g = np.zeros((k, n))

    for i in range(k):
        g[i] = pi[i] * pdf(X, m[i], S[i])

    s_tg = np.sum(g, axis=0, keepdims=True)
    g /= s_tg

    return g, np.sum(np.log(s_tg))
