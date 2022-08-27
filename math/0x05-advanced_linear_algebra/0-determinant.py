#!/usr/bin/env python3
"""
Matrix determinant
"""


def Laplace(matrix):
    """
    Using Laplace Expansion method to get the determinant of a matrix.
    (recursion)
    """
    sum = 0
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    else:
        for index in range(len(matrix)):
            Cofactor = Laplace([row[1:] for row in matrix if
                                matrix.index(row) != index])
            if index % 2 == 0:
                sum += matrix[index][0] * Cofactor
            else:
                sum -= matrix[index][0] * Cofactor
    return sum


def determinant(matrix):  # matrix shape: nxn
    """
    Function that calculates the determinant of a matrix:
        - matrix is a list of lists whose determinant should be calculated.
        - If matrix is not a list of lists, raise a TypeError with the
message matrix must be a list of lists.
        - If matrix is not square, raise a ValueError with the message
matrix must be a square matrix.
        - The list [[]] represents a 0x0 matrix.
        - Returns: the determinant of matrix.
    """

    if type(matrix) is not list or not matrix:
        raise TypeError('matrix must be a list of lists')
    if matrix == [[]]:  # 0x0 matrix
        return 1
    col = len(matrix)
    for row in matrix:
        if type(row) is not list:
            raise TypeError('matrix must be a list of lists')
        if len(row) != col:
            raise ValueError('matrix must be a square matrix')
    if col == 1:
        return matrix[0][0]

    return Laplace(matrix)
