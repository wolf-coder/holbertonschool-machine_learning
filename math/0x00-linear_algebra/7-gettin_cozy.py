#!/usr/bin/env python3
"""
simply a module documentation line.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Function that concatenates two matrices along a specific axis:
    - You can assume that mat1 and mat2 are 2D matrices containing ints/floats
    - You can assume all elements in the same dimension are of the
    same type/shape
    - You must return a new matrix
    - If the two matrices cannot be concatenated, return None
    """
    if axis:
        if len(mat1) == len(mat2):
            return [x + y for x, y in zip(mat1, mat2)]
    else:
        if len(mat1[0]) == len(mat2[0]):
            return mat1 + mat2
    return None
