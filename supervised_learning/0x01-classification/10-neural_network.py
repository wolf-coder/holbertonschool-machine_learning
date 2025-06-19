#!/usr/bin/env python3
"""
Neural Network
"""
import numpy as np


class NeuralNetwork():
    """
    Neural network with one hidden layer for binary classification.
    """

    def __init__(self, nx, nodes):
        """
        Initialize the neural network.

        Parameters
        ----------
        nx : int
            Number of input features.
            Must be a positive integer.

        nodes : int
            Number of nodes in the hidden layer.
            Must be a positive integer.

        Raises
        ------
        TypeError
            If `nx` is not an integer.
        ValueError
            If `nx` is less than 1.
        TypeError
            If `nodes` is not an integer.
        ValueError
            If `nodes` is less than 1.

        Attributes
        ----------
        W1 : np.ndarray of shape (nodes, nx)
            Weights for the hidden layer,
        initialized with a normal distribution.

        b1 : np.ndarray of shape (nodes, 1)
            Biases for the hidden layer, initialized with zeros.

        A1 : float
            Activated output for the hidden layer, initialized to 0.

        W2 : np.ndarray of shape (1, nodes)
            Weights for the output neuron,
            initialized with a normal distribution.

        b2 : float
            Bias for the output neuron, initialized to 0.

        A2 : float
            Activated output for the output neuron, initialized to 0.

        Notes
        -----
        The attributes are private and can be accessed via
        the corresponding property methods:
        - W1, b1, A1: Weights, bias, and activation from the hidden layer.
        - W2, b2, A2: Weights, bias, and activation from the output neuron.

        """

        #  nx: verification|assignment
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx

        #  nodes: verification|assignment
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.nodes = nodes
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros(shape=(nodes, 1))
        self.__A1 = 0

        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.

        Parameters
        ----------
        X : np.ndarray of shape (nx, m)
            Input data where:
            - nx is the number of input features.
            - m is the number of examples.

        Returns
        -------
        tuple
            A tuple containing:
            - A1 : np.ndarray of shape (nodes, m)
                Activated output of the hidden layer.
            - A2 : np.ndarray of shape (1, m)
                Activated output of the output neuron (final predictions).

        Notes
        -----
        This method updates the internal state:
        - __A1 : stores the hidden layer activation.
        - __A2 : stores the output neuron activation.
        The activation function used in both layers is the sigmoid function.
        """

        # first layer work
        Z1 = (self.__W1 @ X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))

        # second(=final) layer work
        Z2 = (self.__W2 @ self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return (self.__A1, self.__A2)
