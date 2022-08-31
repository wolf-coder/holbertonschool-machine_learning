#!/usr/bin/env python3
"""
definiteness of a matrix.
"""


def definiteness(matrix):
    import numpy as np
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
    if len(Shape) != 2\
       or (Shape[0] != Shape[1])\
       or not np.array_equal(matrix, matrix.transpose()):  # Not a valid matrix
        return None

    EigenValues = np.linalg.eigvals(matrix)
    Str = ""
    if np.all(EigenValues >= 0):
        Str = 'Positive definite'
        if np.any(EigenValues == 0):
            Str = 'Positive semi-definite'
        return Str
    if np.all(EigenValues <= 0):
        Str = 'Negative definite'
        if np.any(EigenValues == 0):
            Str = 'Negative semi-definite'
        return Str

    return 'Indefinite'
