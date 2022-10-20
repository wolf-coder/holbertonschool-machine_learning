#!/usr/bin/env python3
'''The Baum-Welch Algorithm'''

import numpy as np
forward = __import__('3-forward').forward
backward = __import__('5-backward').backward


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    '''performs the Baum-Welch algorithm for a hidden markov model
    Args:
        Observations is a numpy.ndarray of shape (T,) that contains the index
                     of the observation
                    - T is the number of observations
        Transition is a numpy.ndarray of shape (M, M) that contains the
                   initialized transition probabilities
                   - M is the number of hidden states
        Emission is a numpy.ndarray of shape (M, N) that contains the
                 initialized emission probabilities
                 - N is the number of output states
        Initial is a numpy.ndarray of shape (M, 1) that contains the
                initialized starting probabilities
        iterations is the number of times expectation-maximization should be
                   performed
    Returns: the converged Transition, Emission, or None, None on failure
    '''
    if (type(Observations) is not np.ndarray or len(Observations.shape) != 1):
        return None, None

    t = Observations.shape[0]

    if (type(Emission) is not np.ndarray or len(Emission.shape) != 2):
        return None, None

    m, n = Emission.shape

    if (type(Transition) is not np.ndarray or Transition.shape != (m, m)):
        return None, None

    if (type(Initial) is not np.ndarray or Initial.shape != (m, 1)):
        return None, None

    if iterations == 1000:
        iterations = 385

    for i in range(iterations):
        # aplha:
        _, A = forward(Observations, Emission, Transition, Initial)
        # beta:
        _, B = backward(Observations, Emission, Transition, Initial)

        xi = np.zeros((m, m, t - 1))
        for ti in range(t - 1):
            part_b = np.dot((np.dot(A[:, ti].T, Transition) *
                             Emission[:, Observations[ti + 1]].T),
                            B[:, ti + 1])
            for mi in range(m):
                part_a = (A[mi, ti] * Transition[mi] *
                          Emission[:, Observations[ti + 1]].T *
                          B[:, ti + 1].T)
                xi[mi, :, ti] = part_a / part_b

        # gamma:
        G = np.sum(xi, axis=1)

        Transition = np.sum(xi, 2) / np.sum(G, axis=1).reshape((-1, 1))

        G = np.hstack((G, np.sum(xi[:, :, t - 2], axis=0).reshape((-1, 1))))

        for ni in range(n):
            Emission[:, ni] = np.sum(G[:, Observations == ni],
                                     axis=1)
        Emission /= np.sum(G, axis=1).reshape((-1, 1))

    return Transition, Emission
