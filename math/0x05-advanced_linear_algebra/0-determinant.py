#!/usr/bin/env python3
"""
Matrix determinant
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
    """
    if len(matrix) == 0 or\
       any([(False if type(el) == list else True) for el in matrix]):
        raise TypeError("matrix must be a list of lists")

    if matrix == [[]]:  # [[]] is a 0x0 matrix => (det = 1)
        return 1
    shape = matrix_shape(matrix)
    if shape[0] != shape[1]:
        raise ValueError('matrix must be a square matrix')

    if matrix_shape(matrix) == [1, 1]:  # 1x1 matrix
        return matrix[0][0]
    """
    if type(matrix) is not list or not matrix:
        raise TypeError('matrix must be a list of lists')
    if matrix == [[]]:
        return 1
    m = len(matrix)
    for row in matrix:
        if type(row) is not list:
            raise TypeError('matrix must be a list of lists')
        if len(row) !=  m:
            raise ValueError('matrix must be a square matrix')
    if m == 1:
        return matrix[0][0]
    
    return Laplace(matrix)
