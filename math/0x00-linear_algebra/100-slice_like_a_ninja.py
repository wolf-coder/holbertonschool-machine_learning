#!/usr/bin/env python3
"""
Module with slicing medhod
"""


def np_slice(matrix, axes={}):
    """
    Method that slices a matrix along specific axes:
       - .You can assume that matrix is a numpy.ndarray
       - .You must return a new numpy.ndarray
       - .axes is a dictionary where the key is an axis to slice along
 and the value is a tuple representing the slice to make along that axis
       - .You can assume that axes represents a valid slice
    """
    Slice = ((slice(*axes[i]) if i in axes.keys() else slice(None, None, None))
             for i in range(matrix.ndim))
    return matrix[tuple(Slice)]
