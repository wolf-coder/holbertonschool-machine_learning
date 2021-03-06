#!/usr/bin/env python3
"""
Normal distribution
"""


class Normal():
    """
    Work with normal distribution.
    """

    pi = 3.1415926536
    e = 2.7182818285

    def __init__(self, data=None, mean=0., stddev=1.):
        """
Class contructor def __init__(self, data=None, mean=0., stddev=1.):
    data is a list of the data to be used to estimate the distribution
    mean is the mean of the distribution
    stddev is the standard deviation of the distribution
    Sets the instance attributes mean and stddev
        Saves mean and stddev as floats
    If data is not given (i.e. None (be careful: not data has not the
    same result as data is None))
        Use the given mean and stddev
        If stddev is not a positive value or equals to 0, raise a
ValueError with the message stddev must be a positive value.
    If data is given:
        Calculate the mean and standard deviation of data
        If data is not a list, raise a TypeError with the message
    'data must be a list'
        If data does not contain at least two data points, raise
    a ValueError with the message data must contain multiple values
        """
        self.mean = float(mean)
        self.stddev = float(stddev)
        if data is None:
            if self.stddev <= 0:
                raise ValueError("stddev must be a positive value")
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            self.stddev = ((sum([(elm - self.mean) ** 2 for elm in data]) /
                            len(data))
                           ** 0.5)

    def z_score(self, x):
        """
        Instance method that calculates the z-score of a given x-value
        x is the x-value
        Returns the z-score of x
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        instance method that calculates the x-value of a given z-score
        z is the z-score
        Returns the x-value of z
        """
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x-value
        x is the x-value
        Returns the PDF value for x
        """
        return Normal.e ** (-0.5 * self.z_score(x) ** 2)\
            / (self.stddev * (2 * Normal.pi) ** 0.5)

    def Error_F(z):
        """
        The Error function's Mclaurin series
        """
        return (2 / Normal.pi ** 0.5) * \
            (z - pow(z, 3)/3 + pow(z, 5)/10 - pow(z, 7)/42 + pow(z, 9)/216)

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given x-value
        x is the x-value
        Returns the CDF value for x
        """
        return 0.5 * (1 + Normal.Error_F(self.z_score(x) / pow(2, 1/2)))
