#!/usr/bin/env python3
"""
Creating a pd.DataFrame from a np.ndarray
"""
import numpy as np
import pandas as pd


def from_numpy(array):
    """
    Function that creates a pd.DataFrame from a np.ndarray:
        + array is the np.ndarray from which you should create the pd.DataFrame
        + The columns of the pd.DataFrame should be labeled inalphabetical
order and capitalized. There will not be more than 26 columns.
        + Returns: the newly created pd.DataFrame
    """
    Alpha_List = [chr(i+65) for i in range(array.shape[1])]
    return pd.DataFrame(array, columns=Alpha_List)
