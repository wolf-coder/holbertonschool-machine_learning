#!/usr/bin/env python3
"""
exponential distribution
"""


class Exponential:
    """
    Create a class Exponential that represents an
    exponential distribution
    """

    pi = 3.1415926536
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """
        constructor method
        """
        lambtha = float(lambtha)
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = lambtha
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            """ maximum likelihood estimate for the rate parameter lambtha
            """
            self.lambtha = len(data) / sum(data)
