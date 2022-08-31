#!/usr/bin/env python3
""" Matrix definiteness """

import numpy as np


def definiteness(matrix):
    """
    Function that calculates the definiteness of a matrix:
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    Shape = matrix.shape
    if len(Shape) != 2 or (Shape[0] != Shape[1]):
        return None
    Transpose = matrix.transpose()
    if not np.array_equal(matrix, Transpose):
        return None

    EigenValues, _ = np.linalg.eig(matrix)
    if all(EigenValues > 0):
        return 'Positive definite'
    elif all(EigenValues >= 0):
        return 'Positive semi-definite'
    elif all(EigenValues < 0):
        return 'Negative definite'
    elif all(EigenValues <= 0):
        return 'Negative semi-definite'
    else:
        return 'Indefinite'
