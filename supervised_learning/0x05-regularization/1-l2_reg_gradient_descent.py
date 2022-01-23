#!/usr/bin/env python3
"""
L2 regularization module.
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Gradient Descent with L2 regularization
    """
    size = Y.shape[1]
    W_Dim = weights.copy()
    for i in range(L, 0, -1):
        A_prime = cache['A' + str(i)]
        if i == L:
            dz = A_prime - Y
        else:
            dz = np.matmul(W_Dim['W' + str(i + 1)].T, dz) *\
                (1 - (A_prime * A_prime))
        l2 = (lambtha * W_Dim['W' + str(i)]) / size
        dw = 1 / size * np.matmul(dz, cache['A' + str(i - 1)].T) + l2
        db = 1 / size * np.sum(dz, axis=1, keepdims=True)
        weights["W" + str(i)] = W_Dim['W' + str(i)] - alpha * dw
        weights["b" + str(i)] = W_Dim['b' + str(i)] - alpha * db
