#!/usr/bin/env python3

import matplotlib.pyplot as plt
expectation_maximization = __import__('8-EM').expectation_maximization

def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
Function that finds the best number of clusters for a GMM using the Bayesian Information Criterion:

    X is a numpy.ndarray of shape (n, d) containing the data set
    kmin is a positive integer containing the minimum number of clusters to check for (inclusive)
    kmax is a positive integer containing the maximum number of clusters to check for (inclusive)
        If kmax is None, kmax should be set to the maximum number of clusters possible
    iterations is a positive integer containing the maximum number of iterations for the EM algorithm
    tol is a non-negative float containing the tolerance for the EM algorithm
    verbose is a boolean that determines if the EM algorithm should print information to the standard output
    You should use expectation_maximization from the previous code.
    You may use at most 1 loop
    Returns: best_k, best_result, l, b, or None, None, None, None on failure
        best_k is the best value for k based on its BIC
        best_result is tuple containing pi, m, S
            pi is a numpy.ndarray of shape (k,) containing the cluster priors for the best number of clusters
            m is a numpy.ndarray of shape (k, d) containing the centroid means for the best number of clusters
            S is a numpy.ndarray of shape (k, d, d) containing the covariance matrices for the best number of clusters
        l is a numpy.ndarray of shape (kmax - kmin + 1) containing the log likelihood for each cluster size tested
        b is a numpy.ndarray of shape (kmax - kmin + 1) containing the BIC value for each cluster size tested
            Use: BIC = p * ln(n) - 2 * l
            p is the number of parameters required for the model : number-of-parameters-to-be-learned-in-k-guassian-mixture-model
            n is the number of data points used to create the model
            l is the log likelihood of the model
    """
    if (
            not isinstance(X, np.ndarray) or
            len(X.shape) != 2 or
            not isinstance(kmin, int) or
            kmin < 1
            ):
        return (None, None, None, None)
    sample_count, dimention_count = X.shape
    if kmax is None:
        kmax = sample_count
    if (
            not isinstance(kmax, int) or
            kmax < 1 or
            kmax < kmin + 1 or
            not isinstance(iterations, int) or
            iterations < 1 or
            not isinstance(tol, float) or
            tol < 0 or
            not isinstance(verbose, bool)
    ):
        return (None, None, None, None)

    results = []
    log_likelihoods = []
    BICs = []
    for cluster_count in range(kmin, kmax + 1):
        priors, centroids, covariances, responsibilities, log_likelihood = \
            expectation_maximization(
                X, cluster_count, iterations, tol, verbose)
        results.append((priors, centroids, covariances))
        log_likelihoods.append(log_likelihood)
        parameter_count = (
            cluster_count * (dimention_count + 2) * (dimention_count + 1) / 2
            - 1
        )
        BICs.append(
             np.log(sample_count) * parameter_count - 2 * log_likelihood)

    best_index = np.argmin(BICs)
    best_cluster_count = kmin + best_index
    best_parameters = results[best_index]

    return best_cluster_count, best_parameters, np.array(log_likelihoods),\
        np.array(BICs)
