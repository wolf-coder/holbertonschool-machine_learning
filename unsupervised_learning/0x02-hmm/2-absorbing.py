#!/usr/bin/env python3
"""
Determine if a markov chain is absorbing.
"""
import numpy as np


def absorbing(P):
    """
function that determines if a markov chain is absorbing:p

    P is a is a square 2D numpy.ndarray of shape (n, n) representing the
STANDARD TRANSITION MATRIX
        P[i, j] is the probability of transitioning from state i to state j
        n is the number of states in the markov chain
    Returns: True if it is absorbing, or False on failure
    """
    if type(P) is not np.ndarray:
        return False
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        return False

    if not np.allclose(np.sum(P, axis=1), 1):  # Check Markov or not
        return None

    n = P.shape[0]

    Diag = np.diag(P)  # get the matrix diagonal.

    if np.all(Diag != 1):  # None steady state.
        return False

    if np.all(Diag == 1):  # all Steady state.
        return True

    """
    Get F from the limiting transitioning matrix : If F is Singular then P is
not absorbing
    """
    n = P.shape[0]
    # Get I Diag shape
    Diag = np.diag(P)
    I_diag_len = np.count_nonzero(Diag == 1)

    # Get Q
    Q = P[slice(I_diag_len, n), slice(I_diag_len, n)]
    # F = (I - Q).inverse
    try:
        F = np.linalg.inv(np.eye(Q.shape[0]) - Q)
    except np.linalg.LinAlgError:  # singular matrix Error (determinant = ZERO)
        return False
    return True
