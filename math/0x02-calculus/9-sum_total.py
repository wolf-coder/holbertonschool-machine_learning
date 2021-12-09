#!/usr/bin/env python3
"""
Simple doc :)
"""


def summation_i_squared(n):
    """
    function that calculate the sequence sigma (i^2 ) from 1 to n
    """
    if type(n) is not int or n < 1:
        return None
    return int((n * (n + 1) * (2 * n + 1)) / 6)
