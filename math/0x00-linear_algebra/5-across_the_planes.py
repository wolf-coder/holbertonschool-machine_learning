#!/usr/bin/env python3
"""
simply a module documentation line.
"""


def add_matrices2D(mat1, mat2):
    """
    function that adds two matrices element-wise:
    - You can assume that mat1 and mat2 are 2D matrices
    containing ints/floats.
    - You can assume all elements in the same dimension
    are of the same type/shape.
    - You must return a new matrix
    - If mat1 and mat2 are not the same shape, return None
    """

    """
    Avoiding using function matrix_shape under the module 2-size_me_please,
    as it cannot handle empty dimension: exp:
    mat1 = [[], [3]]
    mat2 = [[], [7]]
    """
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    else:
        return [[x + y for x, y in zip(a, b)] for a, b in zip(mat1, mat2)]
