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

    def forward_prop(self, X) -> np.ndarray:
        """
        Calculates the forward propagation of the neuron.

        Parameters
        ----------
        X : np.ndarray of shape (nx, m)
            Input data:
            - nx is the number of input features
            - m is the number of examples

        Returns
        -------
        np.ndarray of shape (1, m)
            The activated output of the neuron (sigmoid).
        """
        z = (self.__W @ X) + self.__b
        # activation_output =
        output = 1 / (1 + np.exp(-z))
        self.__A = output
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.

        Parameters
        ----------
        Y : np.ndarray of shape (1, m)
            Correct labels for the input data.

        A : np.ndarray of shape (1, m)
            Activated output of the neuron for each example.

        Returns
        -------
        float
            The logistic regression cost.
        Notes
        -----
        epsilon = 1e-8: a standard practice for avoiding log(0) while
        keeping output close to expected.
        """
        # Add a small epsilon for numerical stability to avoid log(0)
        epsilon = 1e-8
        m = Y.shape[1]
        loss_row = Y * np.log(A + epsilon) + (1 - Y) * np.log(1 - A + epsilon)
        cost = -np.sum(loss_row) / m
        return cost
