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
        """

        #  nx: verification|assignment
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be positive integer")
        self.nx = nx

        #  nodes: verification|assignment
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be positive integer")
        self.nodes = nodes
        self.W1 = np.random.normal(size=(nodes, nx))
        self.b1 = np.zeros(shape=(nodes, 1))
        self.A1 = 0

        self.W2 = np.random.normal(size=(1, nodes))
        self.b2 = 0
        self.A2 = 0
