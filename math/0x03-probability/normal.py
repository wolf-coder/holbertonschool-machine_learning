#!/usr/bin/env python3
"""
Normal distribution
"""


class Normal():
    """Documentation for Normal

    """
    def __init__(self, data=None, mean=0., stddev=1.):
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
