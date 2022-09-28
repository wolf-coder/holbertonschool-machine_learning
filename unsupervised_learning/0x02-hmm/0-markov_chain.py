#!/usr/bin/env python3
"""
Probability determination of a markov chain being in a particular state
after a specified number of iterations.
"""
import numpy as np


def markov_chain(P, s, t=1):
    """
Function that determines the probability of a markov chain being in a particular stateafter a specified number of iterations:

    P is a square 2D numpy.ndarray of shape (n, n) representing the transition matrix
        P[i, j] is the probability of transitioning from state i to state j
        n is the number of states in the markov chain
    s is a numpy.ndarray of shape (1, n) representing the probability of starting in each state
    t is the number of iterations that the markov chain has been through
    Returns: a numpy.ndarray of shape (1, n) representing the probability of being in a specific state after t iterations, or None on failure
    """
    for i in range(t):
        s = s @ P

    return s
