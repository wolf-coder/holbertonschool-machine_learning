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
        class constructor
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

    def predict(self, X_s):
        """
         predicts the mean and standard deviation of
         points in a Gaussian process
         """
        sk = self.kernel(X_s, X_s)
        k = self.kernel(self.X, X_s)
        k_inv = np.linalg.inv(self.K)
        mu = k.T.dot(k_inv).dot(self.Y).reshape(-1)
        sigma = (sk - k.T.dot(k_inv).dot(k)).diagonal()
        return mu, sigma

    def update(self, X_new, Y_new):
        """
        updates a Gaussian Process
        """
        self.X = np.append(self.X, X_new).reshape(-1, 1)
        self.Y = np.append(self.Y, Y_new).reshape(-1, 1)
        self.K = self.kernel(self.X, self.X)
