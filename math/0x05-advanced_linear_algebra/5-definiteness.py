#!/usr/bin/env python3
""" 0x05-advanced_linear_algebra """

import numpy as np


def definiteness(matrix):
    """
    calculates the definiteness of a matrix:
    matrix is a numpy.ndarray of shape (n, n) whose definiteness should be
        calculated
    If matrix is not a numpy.ndarray, raise a TypeError with the message
        matrix must be a numpy.ndarray
    If matrix is not a valid matrix, return None
    Return: the string Positive definite, Positive semi-definite, Negative
        semi-definite, Negative definite, or Indefinite if the matrix is
        positive definite, positive semi-definite, negative semi-definite,
        negative definite of indefinite, respectively
    If matrix does not fit any of the above categories, return None
    """

    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    transpose = np.transpose(matrix)
    if not np.array_equal(transpose, matrix):
        return None

    w, _ = np.linalg.eig(matrix)

    if all(w > 0):
        return 'Positive definite'
    elif all(w >= 0):
        return 'Positive semi-definite'
    elif all(w < 0):
        return 'Negative definite'
    elif all(w <= 0):
        return 'Negative semi-definite'
    else:
        return 'Indefinixste'
