#!/usr/bin/env python3
"""
concatenating two matrices
"""


def matrix_shape(arg):
    """
    mehtod that calculates the shape of a matrix:
    - You can assume all elements in the same dimension are
        of the same type/shape.
    - The shape should be returned as a list of integers.
    """

    shape = [len(arg)]
    if len(arg) != 0:
        while type(arg[0]) == list:
            shape.append(len(arg[0]))
            arg = arg[0]
        return shape
    else:
        return [0]


def cat_matrices(mat1, mat2, axis=0):
    """
- A function concatenates two matrices along a specific axis:
    . You can assume that mat1 and mat2 are matrices containing ints/floats
    . You can assume all elements in the same dimension are
of the same type/shape
    . You must return a new matrix
    . If you cannot concatenate the matrices, return None
    . You can assume that mat1 and mat2 are never empty
    """
    matrix = []

      if type(mat1) is not type(mat2) or type(mat1) is not list:
        return None
    if axis == 0:
        mat1_Shape = matrix_shape(mat1)
        mat2_Shape = matrix_shape(mat2)
        if len(mat1_Shape) != len(mat2_Shape) or (len(mat1_Shape) > 0 and mat1_Shape[1:] != mat2_Shape[1:]):
            return None
        matrix = mat1 + mat2
    else:
        if len(mat1) != len(mat2):
            return None
        for i in range(len(mat1)):
            m = cat_matrices(mat1[i], mat2[i], axis=(axis - 1))
            if m is None:
                return None
            matrix.append(m)
    return matrix
