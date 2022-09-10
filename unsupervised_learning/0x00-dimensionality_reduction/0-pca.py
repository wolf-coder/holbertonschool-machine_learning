"""
Principal components analysis
"""

import numpy as np


def pca(X, var=0.95):
    """
    function that performs PCA on a dataset:

        X is a numpy.ndarray of shape (n, d) where:
            n is the number of data points
            d is the number of dimensions in each point
            all dimensions have a mean of 0 across all data points
        var is the fraction of the variance that the PCA transformation
should maintain
        Returns: the weights matrix, W, that maintains var fraction
of X‘s original variance
        W is a numpy.ndarray of shape (d, nd) where nd is the new
dimensionality of the transformed X
    """
    #  get the covariance matrix Sigma
    Sigma = np.cov(X)

    #  Perform [U, S, V] = svd(Sigma)
    [U, S, V] = np.linalg.svd(X)

    #  Get Cumulative Sum_variance_S array
    cum_variance_S_array = np.cumsum(S)

    """
    Get the the minimum number of  dimentions=K needed that
    garanties a 0.95 variance and return its corresponding returned matrix
    """
    Total_variance = cum_variance_S_array[-1]
    for k, el in enumerate(cum_variance_S_array):
        if el/Total_variance >= var:
            return V[:k + 1].T
