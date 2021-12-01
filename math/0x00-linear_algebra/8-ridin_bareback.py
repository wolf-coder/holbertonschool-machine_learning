#!/usr/bin/env python3
"""
simply a module documentation line.
"""


def mat_mul(mat1, mat2):
    """
    that performs matrix multiplication:
    - You can assume that mat1 and mat2 are 2D matrices containing ints/floats.
    - You can assume all elements in the same dimension
    are of the same type/shape.
    - You must return a new matrix.
    - If the two matrices cannot be multiplied, return None.
    """
    def Calcul(L, T):
        """
        Function that caluculate a row by column
        as a part of matrix multiplication.
        """
        return L[0] * T[0] + L[1] * T[1]

    if len(mat1[0]) != len(mat2):
        return None
    else:
        return [[Calcul(row, col) for col in zip(*mat2)] for row in mat1]
