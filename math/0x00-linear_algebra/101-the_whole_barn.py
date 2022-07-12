#!/usr/bin/env python3
"""
Recursively construct a new sum of two matrices.
"""


def add_matrices(mat1, mat2):
    """
- A function that adds two matrices:

    .You can assume that mat1 and mat2 are matrices containing ints/floats
    .You can assume all elements in the same dimension are of the same type/shape
    .You must return a new matrix
    .If matrices are not the same shape, return None
    .You can assume that mat1 and mat2 will never be empty
"""
    try:
        if (len(mat1) != len(mat2)):
            return None
        New_mat = []
        for row1, row2 in zip(mat1, mat2):
            New_row = add_matrices(row1, row2)
            if New_row is None:
                return None
            New_mat.append(New_row)
        return New_mat
    except :
        """ reached the top of the matrix, """
        return mat1 + mat2
