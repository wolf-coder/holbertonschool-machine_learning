#!/usr/bin/env python3
"""
Neuron doc
"""
import numpy as np


class Neuron:
    """
    class Neuron that defines a single neuron performing binary classification
    """

    def __init__(self, nx) -> None:
        """
        Initialize a single neuron for binary classification.

        Parameters
        ----------
        nx : int
            The number of input features to the neuron.
            Must be a positive integer.

        Raises
        ------
        TypeError
            If `nx` is not an integer.

        ValueError
            If `nx` is less than 1.

        Attributes
        ----------
        W : np.ndarray of shape (1, nx)
            The weights vector, accessible via the property `W`.

        b : float
            The bias, accessible via the property `b`.

        A : float
            The activated output, accessible via the property `A`.
        """

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A
