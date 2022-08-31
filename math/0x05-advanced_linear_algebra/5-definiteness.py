#!/usr/bin/env python3
""" definiteness of a matrix """

import numpy as np


def definiteness(matrix):
    """
    - Function that calculates the definiteness of a matrix:
    - matrix is a numpy.ndarray of shape (n, n) whose definiteness
        should be calculated.
    - If matrix is not a numpy.ndarray, raise a TypeError with the message
        matrix must be a numpy.ndarray.
    - If matrix is not a valid matrix, return None.
    - Return: the string:
        * Positive definite,
        * Positive semi-definite,
        * Negative semi-definite,
        * Negative definite, or Indefinite
    - If matrix does not fit any of the above categories, return None
    - You may import numpy as np
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    Shape = matrix.shape
    if len(Shape) != 2 or (Shape[0] != Shape[1]):
        return None
    Transpose = matrix.transpose()
    if not np.array_equal(matrix, Transpose):  # Not a valid matrix
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
