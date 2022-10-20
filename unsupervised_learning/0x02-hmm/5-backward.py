#!/usr/bin/env python3
"""
Performing the backward algorithm for a hidden markov model.
"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    def backward(Observation, Emission, Transition, Initial): that performs the backward algorithm for a hidden markov model:

    Observation is a numpy.ndarray of shape (T,) that contains the index of the observation
        T is the number of observations
    Emission is a numpy.ndarray of shape (N, M) containing the emission probability of a specific observation given a hidden state
        Emission[i, j] is the probability of observing j given the hidden state i
        N is the number of hidden states
        M is the number of all possible observations
    Transition is a 2D numpy.ndarray of shape (N, N) containing the transition probabilities
        Transition[i, j] is the probability of transitioning from the hidden state i to j
    Initial a numpy.ndarray of shape (N, 1) containing the probability of starting in a particular hidden state
    Returns: P, B, or None, None on failure
        Pis the likelihood of the observations given the model
        B is a numpy.ndarray of shape (N, T) containing the backward path probabilities
            B[i, j] is the probability of generating the future observations from hidden state i at time j
    """
    try:
        T = Observation.shape[0]
        N = Transition.shape[0]

        B = np.zeros((N, T))
        B[:, T - 1] = np.ones((N))

        for t in range(T - 2, -1, -1):
            for j in range(N):
                B[j, t] = ((B[:, t + 1] * Emission[:, Observation[t + 1]]).dot(Transition[j, :]))

        Pis = np.sum(np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0]))
        return Pis, B
    except Exception:
        return None, None
