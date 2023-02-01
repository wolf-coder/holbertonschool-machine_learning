#!/usr/bin/env python3
"""
POLICY
"""
import numpy as np


def policy(matrix, weight):
    """
    Computes To Policy
    """
    policy = np.exp(matrix.dot(weight)) / np.sum(np.exp(matrix.dot(weight)))
    return policy
