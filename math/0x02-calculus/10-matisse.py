#!/usr/bin/env python3
"""
derive the poly
"""


def poly_derivative(poly):
    """
    calculates the derivative of a polynomial.
    the index of the list represents the power of x
    that the coefficient belongs to.
    """
    if type(poly) is not list or len(poly) == 0:
        return None
    if len(poly) == 1:
        return [0]
    return [index * coefficient
            for index, coefficient in enumerate(poly)
            if index]
