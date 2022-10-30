#!/usr/bin/env python3
"""
Gaussian Process
"""
import numpy as np


class GaussianProcess:
    """
    class GaussianProcess that represents a noiseless 1D Gaussian process
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        C=class constructor
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        calculates the covariance kernel matrix between two matrices
        """
        x1_norm = np.sum(X1**2, 1)
        x2_norm = np.sum(X2**2, 1)
        k0 = x1_norm.reshape(-1, 1) + x2_norm - 2 * np.dot(
            X1, X2.T)
        return (self.sigma_f ** 2) * np.exp((-1 / (2 * self.l ** 2)) * k0)
