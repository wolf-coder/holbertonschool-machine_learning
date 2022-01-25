#!/usr/bin/env python3
"""
Poinsson distribution
"""


class Poisson:
    """Documentation for Poisson
    Class Poisson that represents a poisson distribution.
    """

    pi = 3.1415926536
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """
        - data is a list of the data to be used to estimate the distribution
lambtha is the expected number of occurences in a given time frame
Sets the instance attribute lambtha.

        - If data is not given, (i.e. None (be careful: not data has
        not the same result as data is None)):
            * Use the given lambtha.
            * If lambtha is not a positive value or equals to 0,
raise a ValueError with the message lambtha must be a positive value.

        - If data is given:
            * Calculate the lambtha of data.
            * If data is not a list, raise a TypeError with the message
        data must be a list
            * If data does not contain at least two data points,
raise a ValueError with the message data must contain multiple values
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = lambtha
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            """
            Using the maximum likehood Estimate method to estimate
the parameter lambtha.
            """
            self.lambtha = sum(data)/len(data)

    def Factorial(n):
        """
        calculate factorial
        """
        if n == 0 or n == 1:
            return 1
        k = 1
        for i in range(1, n + 1):
            k *= i
        return k

    def pmf(self, k):
        """
    Calculates the value of the PMF for a given number of “successes”
    k is the number of “successes”
        If k is not an integer, convert it to an integer
        If k is out of range, return 0
    Returns the PMF value for k
        """
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        return (pow(self.lambtha, k) * pow(Poisson.e, -self.lambtha)) /\
            Poisson.Factorial(k)

    def cdf(self, k):
        """
    Calculates the value of the CDF for a given number of “successes”.
    k is the number of “successes”.
        If k is not an integer, convert it to an integer.
        If k is out of range, return 0.
    Returns the CDF value for k.
        """
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        return sum(
            map(self.pmf, range(k + 1))
        )
