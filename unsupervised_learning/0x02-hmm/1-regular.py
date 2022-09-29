#!/usr/bin/env python3
"""
Determination of the steady state probabilities of a regular markov chain.
"""
import numpy as np

def regular(P):
    """
Function that determines the steady state probabilities of a regular markov chain:

    P is a is a square 2D numpy.ndarray of shape (n, n) representing the transition matrix
        P[i, j] is the probability of transitioning from state i to state j
        n is the number of states in the markov chain
    Returns: a numpy.ndarray of shape (1, n) containing the steady state probabilities, or None on failure
    """
    
    n= P.shape[0]
    d=P.shape[1]
    if type(P) is not np.ndarray or P.ndim!=2 or n!= d :
        return None 
    if not np.allclose(np.sum(P, axis=1), 1):
        return None
    n = P.shape[0]
    q = (P-np.eye(n))
    Ones = np.ones(n)
    q = np.c_[q,Ones]
    
    QxQ = np.dot(q, q.T)
    try: # Can't solve it as QxQ is singular or not squarred
        steady_state_prob = np.linalg.solve(QxQ,Ones)
    except np.linalg.LinAlgError:
        return None
    return steady_state_prob.reshape(1, n)
