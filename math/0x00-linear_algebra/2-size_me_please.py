#!/usr/bin/env python3
"""
simply a module documentation line
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
