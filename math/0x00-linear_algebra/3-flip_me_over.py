#!/usr/bin/env python3
"""
simply a module documentation line.
"""


def matrix_transpose(matrix):
    """
    Function that returns the transpose of a 2D matrix, matrix:
    - You must return a new matrix
    - You can assume that matrix is never empty.
    - You can assume all elements in the same dimension
    are of the same type/shap.
    """
    return [list(row) for row in zip(*matrix)]
