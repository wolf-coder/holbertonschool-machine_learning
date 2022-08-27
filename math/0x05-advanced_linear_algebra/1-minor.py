#!/usr/bin/env python3
"""
Matrix Minor
"""

matrix_shape = __import__("0-determinant").matrix_shape
Laplace = __import__("0-determinant").Laplace


def minor(matrix):
    """
    Function that calculates the minor matrix of a matrix:

        - matrix is a list of lists whose minor matrix should be calculated.
        - If matrix is not a list of lists, raise a TypeError with the message
matrix must be a list of lists.
        - If matrix is not square or is empty, raise a ValueError with the
message matrix must be a non-empty square matrix.
        - Returns: the minor matrix of matrix.
    """
    if type(matrix) is not list or not matrix:
        raise TypeError('matrix must be a list of lists')
    if matrix == [[]]:  # [[]] is a 0x0 matrix => (minor = 1)
        return 1
    col = len(matrix)
    for row in matrix:
        if type(row) is not list:
            raise TypeError('matrix must be a list of lists')
        if len(row) != col:
            raise ValueError('matrix must be a square matrix')

    if matrix_shape(matrix) == [1, 1]:  # 1x1 matrix
        return 1

    # 2x2 matrix
    if matrix_shape(matrix) == [2, 2]:  # 1x1 matrix
        return [[matrix[1][1], matrix[1][0]], [matrix[0][1], matrix[0][0]]]

    #  nxn matrix
    M = []
    for i in range(len(matrix)):
        R = []
        for j in range(len(matrix)):
            Sub_matrix = [[row[J] for J in range(len(matrix)) if J != j]
                          for k, row in enumerate(matrix) if k != i]
            R.append(Laplace(Sub_matrix))
        M.append(R)
    return M
