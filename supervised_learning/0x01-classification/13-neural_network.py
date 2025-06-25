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

    def cost(self, Y, A):
        """
        - Calculates the cost of the model using logistic regression
        - Y is a numpy.ndarray with shape (1, m) that contains the correct labels for the input data
        - A is a numpy.ndarray with shape (1, m) containing the activated output of the neuron for each example
        - To avoid division by zero errors, please use 1.0000001 - A instead of 1 - A
        - Returns the cost
        """
        SafeOne = 1.0000001 # to prevent log(0) as 0 is not defined for function log
        m = Y.shape[1]
        loss_row = (Y * np.log(A)) + ((1 - Y) * (np.log(SafeOne - A)))
        cost = np.sum(loss_row) / (- m)
        return cost
        

    def evaluate(self, X, Y):
        """
        Evaluate the neural network predictions
        """
        _, OutputA2 = NeuralNetwork.forward_prop(self, X)
        predictions = np.where(OutputA2>=0.5,1,0)
        cost = NeuralNetwork.cost(self,Y,OutputA2)
        return (predictions, cost)


    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
    - Calculates one pass of gradient descent on the neural network
    - X is a numpy.ndarray with shape (nx, m) that contains the input data
        - nx is the number of input features to the neuron
        - m is the number of examples
    - Y is a numpy.ndarray with shape (1, m) that contains the correct labels for the input data
    - A1 is the output of the hidden layer
    - A2 is the predicted output
    - alpha is the learning rate
    - Updates the private attributes __W1, __b1, __W2, and __b2
        """
        m = X.shape[1]
        # calculate the derivatives

        # 1. Compute the gradient of the loss with respect to the output layer's activations:
        dZ2 = self.A2 - Y

        # 2. Compute the gradients with respect to the weights and biases of the second layer:
        dW2 = (dZ2 @ A1.T) / m
        db2 = (np.sum(dZ2,axis=1,keepdims= True))/m

        # 4. Compute the gradient of the loss with respect to the first layer's activations:
        dZ1 = (self.W2.T @ dZ2) * (A1 * (1-A1))

        # 5. Compute the gradients with respect to the weights and biases of the first layer:
        dW1 = (dZ1 @ X.T) / m
        db1 = ( np.sum(dZ1, axis=1, keepdims=True) )/m


        # 3. Update the weights and biases for the second layer:
        self.__W2 -= (alpha * dW2)
        self.__b2 -= (alpha * db2)

        # 6. Update the weights and biases for the first layer:
        self.__W1 -= (alpha * dW1)
        self.__b1 -= (alpha * db1)
