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


def policy_gradient(state, weight):
    """
    Computes Monte-Carlo policy
    """
    p = policy(state, weight)
    action = np.random.choice(len(p[0]), p=p[0])
    p_r = p.reshape(-1, 1)
    d_softmax = (np.diagflat(p_r) - np.dot(p_r, p_r.T))[action, :]
    d_log = d_softmax / p[0, action]
    grad = state.T.dot(d_log[None, :])
    return action, grad
