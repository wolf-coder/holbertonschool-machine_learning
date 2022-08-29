#!/usr/bin/env python3
"""
Matrix Minor
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
        while len(arg) != 0 and type(arg[0]) == list:
            shape.append(len(arg[0]))
            arg = arg[0]
        return shape
    else:
        return [0]


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


def cofactor(matrix):
    """
    - that calculates the cofactor matrix of a matrix:
    - matrix is a list of lists whose cofactor matrix should be calculated.
    - If matrix is not a list of lists, raise a TypeError with the message
        matrix must be a list of lists.
    - If matrix is not square or is empty, raise a ValueError with the message
        matrix must be a non-empty square matrix.
    - Returns: the cofactor matrix of matrix.
    """
    if type(matrix) is not list or not matrix:
        raise TypeError('matrix must be a list of lists')

    col = len(matrix)
    for row in matrix:
        if type(row) is not list:
            raise TypeError('matrix must be a list of lists')
        if len(row) != col:
            raise ValueError('matrix must be a non-empty square matrix')

    Shape = matrix_shape(matrix)
    if Shape == [1, 1]:  # 1x1 matrix
        return [[1]]

    if Shape[0] != Shape[1]:
        raise ValueError('matrix must be a non-empty square matrix')
    # 2x2 matrix
    if matrix_shape(matrix) == [2, 2]:  # 2x2 matrix
        return [[matrix[1][1], - matrix[1][0]], [- matrix[0][1], matrix[0][0]]]

    #  nxn matrix
    M = []
    for i in range(len(matrix)):
        R = []
        for j in range(len(matrix)):
            Sub_matrix = [[row[J] for J in range(len(matrix)) if J != j]
                          for k, row in enumerate(matrix) if k != i]
            R.append(Laplace(Sub_matrix) * ((-1) ** (i+j+2)))
        M.append(R)
    return M


def adjugate(matrix):
    """
    - Calculates the adjugate matrix of a matrix:
    - matrix is a list of lists whose adjugate matrix should be calculated
    - If matrix is not a list of lists, raise a TypeError with the message
        matrix must be a list of lists
    - If matrix is not square or is empty, raise a ValueError with the message
        matrix must be a non-empty square matrix
    - Returns: the adjugate matrix of matrix
    """
    Matrix_Cofactor = cofactor(matrix)
    return [list(el) for el in zip(*Matrix_Cofactor)]
