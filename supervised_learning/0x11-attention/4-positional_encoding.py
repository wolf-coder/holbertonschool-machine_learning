#!/usr/bin/env python3
"""
POSTIONAL ENCODING
"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    positional encoding
    """
    encod = np.zeros((max_seq_len, dm))
    for i in range(max_seq_len):
        for j in range(0, dm, 2):
            encod[i, j] = np.sin(
                i / np.power(10000, (2 * j // 2) / dm))
            encod[i, j + 1] = np.cos(
                i / np.power(10000, (2 * j // 2) / dm))
    return encod
