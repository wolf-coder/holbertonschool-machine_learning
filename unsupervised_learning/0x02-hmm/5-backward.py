#!/usr/bin/env python3

import numpy as np



def backward(Observation, Emission, Transition, Initial):
    """
    performs the backward algorithm for a hidden markov model
    """
    if not isinstance(Observation, np.ndarray) or len(Observation.shape) != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None
    if not isinstance(Transition, np.ndarray)\
            or len(Transition.shape) != 2\
            or Transition.shape[0] != Transition.shape[1]:
        return None, None
    if not isinstance(Initial, np.ndarray) or len(Initial.shape) != 2:
        return None, None
    if Emission.shape[0] != Transition.shape[0] != Transition.shape[0] !=\
       Initial.shape[0]:
        return None, None
    if Initial.shape[1] != 1:
        return None, None

    T = Observation.shape[0]
    N = Transition.shape[0]

    beta = np.zeros((N, T))
    beta[:, T - 1] = np.ones((N))

    for t in range(T - 2, -1, -1):
        for j in range(N):
            beta[j, t] = ((beta[:, t + 1]
                           * Emission[:, Observation[t + 1]]).
                          dot(Transition[j, :]))

    likelihood = np.sum(np.sum(Initial.T
                               * Emission[:, Observation[0]]
                               * beta[:, 0]))

    return likelihood, beta
