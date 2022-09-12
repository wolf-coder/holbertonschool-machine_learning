#!/usr/bin/env python3
"""
Function that calculates the likelihood of obtaining this data given
various hypothetical probabilities of developing severe side effects
"""

import numpy as np


def likelihood(x, n, P):
    """
Function that calculates the likelihood of obtaining
this data given various hypothetical probabilities of
developing severe side effects:

    x is the number of patients that develop severe side effects
    n is the total number of patients observed
    P is a 1D numpy.ndarray containing the various hypothetical
probabilities of developing severe side effects
    If n is not a positive integer, raise a ValueError with the message
n must be a positive integer
    If x is not an integer that is greater than or equal to 0,
raise a ValueError with the message x must be an integer that is
greater than or equal to 0
    If x is greater than n, raise a ValueError with the message x
cannot be greater than n
    If P is not a 1D numpy.ndarray, raise a TypeError with
the message P must be a 1D numpy.ndarray
    If any value in P is not in the range [0, 1],
raise a ValueError with the message All values in P must be in the range [0, 1]
    Returns: a 1D numpy.ndarray containing the likelihood
of obtaining the data, x and n, for each probability in P, respectively
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError('n must be a positive integer')
    if not isinstance(x, int) or x < 0:
        raise ValueError('x must be an integer that is greater than or equal to 0')
    if x > n:
        raise ValueError('x cannot be greater than n')
    
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError('P must be a 1D numpy.ndarray')
    if not ((P <= 1).all() and (P <= 1).all()):
        raise ValueError('All values in P must be in the range [0, 1]')

    Fact = np.math.factorial  # Factorial Function
    Pow = np.math.pow
    Comb = Fact(n) / (Fact(x) * Fact(n - x))
    Arr_To_P = [(Comb * Pow(p, x) * Pow(1 - p, n - x)) for p in P]
    return np.array(Arr_To_P)
