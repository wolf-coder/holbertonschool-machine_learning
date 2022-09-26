#!/usr/bin/env python3
"""
Performs the expectation maximization for a GMM.
"""
import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
Function  that performs the expectation maximization for a GMM:

    X is a numpy.ndarray of shape (n, d) containing the data set
    k is a positive integer containing the number of clusters
    iterations is a positive integer containing the maximum number of iterations for the algorithm
    tol is a non-negative float containing tolerance of the log likelihood, used to determine early stopping i.e. if the difference is less than or equal to tol you should stop the algorithm
    verbose is a boolean that determines if you should print information about the algorithm
        If True, print Log Likelihood after {i} iterations: {l} every 10 iterations and after the last iteration
        {i} is the number of iterations of the EM algorithm
        {l} is the log likelihood, rounded to 5 decimal places
    You should use:
        initialize from the previous code.
        expectation from the previous code.
        maximization from the previous code.
    You may use at most 1 loop
    Returns: pi, m, S, g, l, or None, None, None, None, None on failure
        pi is a numpy.ndarray of shape (k,) containing the priors for each cluster
        m is a numpy.ndarray of shape (k, d) containing the centroid means for each cluster
        S is a numpy.ndarray of shape (k, d, d) containing the covariance matrices for each cluster
        g is a numpy.ndarray of shape (k, n) containing the probabilities for each data point in each cluster
        l is the log likelihood of the model
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or int(k) != k or k < 1:
        return None, None, None, None, None
    if not isinstance(iterations, int) or int(iterations) != iterations or iterations < 1:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None, None
    pi, m, S = initialize(X, k)
    Log = None
    for i in range(iterations):
        Posterior, l = expectation(X, pi, m, S)
        if Log is not None and np.abs(l - Log) <= tol:
            if verbose:
                print('Log Likelihood after {} iterations: {}'.format(i, l.round(5)))
            break
        if verbose and i % 10 == 0:
            print('Log Likelihood after {} iterations: {}'.format(i, l.round(5)))
        pi, m, S = maximization(X, Posterior)
        Log = l
    else:
        Posterior, l = expectation(X, pi, m, S)
        if verbose:
            print('Log Likelihood after {} iterations: {}'.format(iterations, l.round(5)))
    return pi, m, S, Posterior, l
