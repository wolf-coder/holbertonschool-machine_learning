#!/usr/bin/env python3
"""
Calculates the most likely sequence of hidden states for a hidden markov model.
"""
import numpy as np

def viterbi(Observation, Emission, Transition, Initial):
    """
    Observation is a numpy.ndarray of shape (T,) that contains the index of the observation
        T is the number of observations
    Emission is a numpy.ndarray of shape (N, M) containing the emission probability of a specific observation given a hidden state
        Emission[i, j] is the probability of observing j given the hidden state i
        N is the number of hidden states
        M is the number of all possible observations
    Transition is a 2D numpy.ndarray of shape (N, N) containing the transition probabilities
        Transition[i, j] is the probability of transitioning from the hidden state i to j
    Initial a numpy.ndarray of shape (N, 1) containing the probability of starting in a particular hidden state
    Returns: path, P, or None, None on failure
        path is the a list of length T containing the most likely sequence of hidden states
        P is the probability of obtaining the path sequence
    """
    try:
        T = Observation.shape[0]
        N = Emission.shape[0]
        viterbi = np.zeros((N, T))
        viterbi[:, 0] = Initial.T * Emission[:, Observation[0]]
        backpointer = np.zeros((N, T))
        for t in range(1, T):
            for s in range(N):
                tr = Transition[:, s]
                em = Emission[s, Observation[t]]
                viterbi[s, t] = np.max(viterbi[:, t - 1] * tr * em)
                backpointer[s, t] = np.argmax(viterbi[:, t - 1] * tr * em)
        path = [np.argmax(viterbi[:, T - 1])]
        for t in range(T - 1, 0, -1):
            path.append(int(backpointer[path[-1], t]))
        path = path[::-1]
        p = np.max(viterbi[:, T - 1])
        return path, p
    except Exception:
        return None, None
