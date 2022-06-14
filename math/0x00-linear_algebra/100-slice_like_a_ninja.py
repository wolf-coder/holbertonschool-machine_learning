#!/usr/bin/env python3
import numpy as np


def np_slice(matrix, axes={}):
    """
    """
    Slice = ((slice(*axes[i]) if i in axes.keys() else slice(None, None, None))
             for i in range(matrix.ndim))
    return matrix[tuple(Slice)]
