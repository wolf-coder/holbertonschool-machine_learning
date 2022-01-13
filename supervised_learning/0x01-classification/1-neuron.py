#!/usr/bin/env python3
"""
Neuron
"""
import numpy as np


class Neuron:
    """
    class Neuron that defines a single neuron performing binary classification
    """

    def __init__(self, nx):
        """
        Initialization constructor
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        __W private attribute getter
        """
        return self.__W

    @property
    def b(self):
        """
        __b private attribute Getter
        """
        return self.__b

    @property
    def A(self):
        """
        __A private attribute Getter
        """
        return self.__A
