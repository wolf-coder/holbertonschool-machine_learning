#!/usr/bin/env python3
"""
simply a module documentation line.
"""
matrix_shape = __import__('2-size_me_please').matrix_shape


def add_arrays(arr1, arr2):
    """
    Write a function def add_arrays(arr1, arr2): that adds two arrays
    element-wise:
    - You can assume that arr1 and arr2 are lists of ints/floats
    - You must return a new list
    - If arr1 and arr2 are not the same shape, return None
    """
    if matrix_shape(arr1) != matrix_shape(arr2):
        return None
    else:
        return [a + b for a, b in zip(arr1, arr2)]
