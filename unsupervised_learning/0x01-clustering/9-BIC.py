#!/usr/bin/env python3
"""
Finding the best number of clusters for a GMM using the Bayesian Information Criterion.
"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization

def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Function that finds the best number of clusters for a GMM using the Bayesian Information Criterion.
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
