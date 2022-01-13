#!/usr/bin/env python3
"""
Representing a binomial distribution:
"""


class Binomial:
    """
    work with Binomial distribution
    """
    def __init__(self, data=None, n=1, p=0.5):
        """
        - Initialization contructor.
        - Using a `method-of-moments` estimator for p and n.
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not 0 < p < 1:
                raise ValueError("p must be greater than 0 and less than 1")
            else:
                self.p = float(p)
                self.n = int(n)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            Expectation = sum(data)/len(data)
            Variance = sum([(Xi - Expectation) ** 2 for Xi in data]) /\
                len(data)
            self.p = 1 - Variance / Expectation
            self.n = round(Expectation / self.p)
            self.p = Expectation / self.n
