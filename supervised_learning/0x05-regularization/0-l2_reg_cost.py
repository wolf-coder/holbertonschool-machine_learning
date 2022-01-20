#!/usr/bin/env python3
"""
L2 Regularization
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization.
    """
    norm = 0
    for k, weight in weights.items():
        if k[0] == 'W':
            norm += np.linalg.norm(weight)
    cost += lambtha / (2 * m) * norm
    return cost
