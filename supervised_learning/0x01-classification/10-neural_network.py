#!/usr/bin/env python3
import numpy as np

class NeuralNetwork():
    """
    Neural network with one hiddden layer.
    """
    def __init__(self, nx, nodes):
        """
    W1: The weights vector for the hidden layer. (node, nx)
    b1: The bias for the hidden layer.(node, 1)
    A1: The activated output for the hidden layer. (node, 1)
    W2: The weights vector for the output neuron. (node, 1)
    b2: The bias for the output neuron . (1)
    A2: The activated output for the output neuron (prediction). (1)
        """
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
    - Calculates the forward propagation of the neural network
    - X is a numpy.ndarray with shape (nx, m) that contains the input data
        - nx is the number of input features to the neuron
        - m is the number of examples
    - Updates the private attributes __A1 and __A2
    - The neurons should use a sigmoid activation function
    - Returns the private attributes __A1 and __A2, respectively
        """
        Z1 = (self.__W1 @ X) + self.__b1 # (node x m)
        A1 = 1 / (1 + np.exp(- Z1))
        self.__A1 = A1 # (node x m)

        Z2 = (self.__W2 @ self.__A1) + self.__b2
        A2 = 1 / (1 + np.exp(- Z2))
        self.__A2 = A2 

        return (self.__A1, self.__A2)

