#!/usr/bin/env python3
"""
Representing a binomial distribution:
"""


class Binomial:
    def __init__(self, data=None, n=1, p=0.5):
        """
        data is a list of the data to be used to estimate the distribution
        n is the number of Bernoulli trials
        p is the probability of a “success”
        Sets the instance attributes n and p
            Saves n as an integer and p as a float
        If data is not given (i.e. None)
            Use the given n and p
            If n is not a positive value, raise a ValueError
with the message n must be a positive value
            If p is not a valid probability,
raise a ValueError with the message p must be greater than 0 and less than 1
        If data is given:
            Calculate n and p from data
            Round n to the nearest integer (rounded,
not casting! The difference is important: int(3.7) is
not the same as round(3.7))
            Hint: Calculate p first and then calculate n.
Then recalculate p. Think about why you would want to do it this way?
            If data is not a list,
raise a TypeError with the message data must be a list
            If data does not contain at least two data points,
raise a ValueError with the message data must contain multiple values
        """
        self.n = int(n)
        self.p = float(p)

        if not data:
            if self.n < 0:
                raise ValueError("n must be a positive value")
            if not 0 < self.p < 1:
                raise ValueError("p must be greater than 0 and less than 1")
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            """
            Using a `method-of-moments` estimator for p and n
            """
            Expectation = sum(data)/len(data)

            Variance = sum([(Xi - Expectation) ** 2 for Xi in data]) /\
                len(data)

            p = 1 - Variance / Expectation
            self.n = round(Expectation / p)
            self.p = Expectation / self.n
