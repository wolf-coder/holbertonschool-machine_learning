#!/usr/bin/env python3
"""
Simple doc :)
"""


def summation_i_squared(n):
    """
    function that calculate the sequence sigma (i^2 ) from 1 to n
    """
    if type(n) is not int:
        return None
    if n == 1:
        return 1
    else:
        return n ** 2 + summation_i_squared(n-1)
