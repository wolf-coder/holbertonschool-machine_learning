#!/usr/bin/env python3
"""
Performing the Baum-Welch algorithm for a hidden markov model.
"""
def backward(Observation, Emission, Transition, Initial):
    """
    def backward(Observation, Emission, Transition, Initial): that
performs the backward algorithm for a hidden markov model:

    Observation is a numpy.ndarray of shape (T,) that contains the
index of the observation
        T is the number of observations
    Emission is a numpy.ndarray of shape (N, M) containing the
emission probability of a specific observation given a hidden state
        Emission[i, j] is the probability of observing j given the
hidden state i
        N is the number of hidden states
        M is the number of all possible observations
    Transition is a 2D numpy.ndarray of shape (N, N) containing the
transition probabilities
        Transition[i, j] is the probability of transitioning from the
hidden state i to j
    Initial a numpy.ndarray of shape (N, 1) containing the
probability of starting in a particular hidden state
    Returns: P, B, or None, None on failure
        Pis the likelihood of the observations given the model
        B is a numpy.ndarray of shape (N, T) containing the backward
path probabilities
            B[i, j] is the probability of generating the future
observations from hidden state i at time j
    """
    try:
        T = Observation.shape[0]
        N = Transition.shape[0]

        B = np.zeros((N, T))
        B[:, T - 1] = np.ones((N))

        for t in range(T - 2, -1, -1):
            for j in range(N):
                B[j, t] = (B[:, t + 1] * Emission[:, Observation[t + 1]]) @\
                    (Transition[j, :])

        Pis = np.sum(np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0]))
        return Pis, B
    except Exception:
        return None, None


def forward(Observation, Emission, Transition, Initial):

    """
    Function that performs the forward algorithm for a hidden markov model:

    Observation is a numpy.ndarray of shape (T,) that contains
the index of the observation
        T is the number of observations
    Emission is a numpy.ndarray of shape (N, M) containing the
emission probability of a specific observation given a hidden state
        Emission[i, j] is the probability of observing j given the
hidden state i
        N is the number of hidden states
        M is the number of all possible observations
    Transition is a 2D numpy.ndarray of shape (N, N) containing
the transition probabilities
        Transition[i, j] is the probability of transitioning
from the hidden state i to j
    Initial a numpy.ndarray of shape (N, 1) containing the probability
of starting in a particular hidden state
    Returns: P, F, or None, None on failure
        P is the likelihood of the observations given the model
        F is a numpy.ndarray of shape (N, T) containing the forward
path probabilities
        F[i, j] is the probability of being in hidden state i at time
j given the previous observations
    """

    T = Observation.shape[0]
    N = Transition.shape[0]
    F = np.ones((N, T))  # create a probability matrix forward[N,T]
    try:
        F[:, 0] = Initial.T * Emission[:, Observation[0]]  # Initialisaton Step
        for t in range(1, T):  # Recursion step.
            for s in range(N):
                tr = Transition[slice(None), s]
                em = Emission[s, Observation[t]]
                F[s, t] = np.sum(F[:, t - 1] * tr * em)
        P = np.sum(F[:, T - 1])  # Termination Step.
        return P, F
    except Exception:
        return None, None


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
