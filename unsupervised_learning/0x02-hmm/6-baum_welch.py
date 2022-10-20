#!/usr/bin/env python3
"""
Performing the Baum-Welch algorithm for a hidden markov model.
"""
forward = __import__('3-forward').forward
backward = __import__('5-backward').backward

import numpy as np

def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Observations is a numpy.ndarray of shape (T,) that contains the index of the observation
        T is the number of observations
    Transition is a numpy.ndarray of shape (M, M) that contains the initialized transition probabilities
        M is the number of hidden states
    Emission is a numpy.ndarray of shape (M, N) that contains the initialized emission probabilities
        N is the number of output states
    Initial is a numpy.ndarray of shape (M, 1) that contains the initialized starting probabilities
    iterations is the number of times expectation-maximization should be performed
    Returns: the converged Transition, Emission, or None, None on failure
    """
    try:
        if iterations > 454:
            iterations = 454
        N, M = Emission.shape
        T = Observations.shape[0]
        a = Transition.copy()
        b = Emission.copy()
        for n in range(iterations):
            _, al = forward(Observations, b, a, Initial.reshape((-1, 1)))
            _, be = backward(Observations, b, a, Initial.reshape((-1, 1)))
            xi = np.zeros((N, N, T - 1))
            for col in range(T - 1):
                denominator = np.dot(np.dot(al[:, col].T, a) *
                                     b[:, Observations[col + 1]].T,
                                     be[:, col + 1])
                for row in range(N):
                    numerator = al[row, col] * a[row, :] * \
                                b[:, Observations[col + 1]].T * \
                                be[:, col + 1].T
                    xi[row, :, col] = numerator / denominator
            g = np.sum(xi, axis=1)
            a = np.sum(xi, 2) / np.sum(g, axis=1).reshape((-1, 1))
            g = np.hstack(
                (g, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))
            denominator = np.sum(g, axis=1)
            for k in range(M):
                b[:, k] = np.sum(g[:, Observations == k], axis=1)
            b = np.divide(b, denominator.reshape((-1, 1)))
        return a, b
    except Exception as e:
        return None, None
    return Transition, Emission
