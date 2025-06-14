#!/usr/bin/env python3
"""
Neuron
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
            The number of input features to the neuron. Must be a positive integer.

        Raises
        ------
        TypeError
            If `nx` is not an integer.

        ValueError
            If `nx` is less than 1.

        Attributes
        ----------
        __W : np.ndarray of shape (1, nx)
            The weights vector, initialized using a normal distribution.

        __b : float
            The bias, initialized to 0.

        __A : float
            The activated output, initialized to 0.
        """

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.W = np.random.normal(size=(1, nx))
        self.b = 0
        self.A = 0
