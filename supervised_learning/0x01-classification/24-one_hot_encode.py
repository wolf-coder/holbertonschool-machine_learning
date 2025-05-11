#!/usr/bin/env python3

import numpy as np
def one_hot_encode(Y, classes):
    """
    Function that converts a numeric label vector into a one-hot matrix:

    Y is a numpy.ndarray with shape (m,) containing numeric class labels
        m is the number of examples
    classes is the maximum number of classes found in Y
    Returns: a one-hot encoding of Y with shape (classes, m), or None on failure
    """

    one_hot = np.zeros((classes, Y.shape[0]))
    one_hot[Y, np.arange(Y.shape[0])] = 1
    return one_hot
