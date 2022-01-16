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
        return Binomial.Factorial(self.n) /\
            (Binomial.Factorial(k) * Binomial.Factorial(self.n - k)) *\
            pow(self.p, k) * pow(1 - self.p, self.n - k)

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of “successes”
        k is the number of “successes”
            If k is not an integer, convert it to an integer
            If k is out of range, return 0
        Returns the CDF value for k
        NOTE: using the pmf method
        """
        if type(k) is not int:
            k = int(k)
        if k > self.n or k < 0:
            return 0
        PMFs_to_k = map(self.pmf, range(0, k + 1))
        return sum(PMFs_to_k)
