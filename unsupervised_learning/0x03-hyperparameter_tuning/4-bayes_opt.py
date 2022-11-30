#!/usr/bin/env python3
"""
Bayesian optimization
"""
from scipy.stats import norm
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """class BayesianOptimization that
    performs Bayesian optimization on a noiseless 1D
    Gaussian process"""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        class constructor
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        xs = np.linspace(bounds[0], bounds[1], num=ac_samples)
        self.X_s = xs.reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        calculates the next best sample location
        """
        mu, sigma = self.gp.predict(self.X_s)
        optm = np.max(self.gp.Y)
        if self.minimize:
            optm = np.min(self.gp.Y)
        ip = optm - mu - self.xsi
        z = ip / sigma
        EI = ip * norm.cdf(z) + sigma * norm.pdf(z)
        X_next = self.X_s[np.argmax(EI)]
        return X_next, EI
