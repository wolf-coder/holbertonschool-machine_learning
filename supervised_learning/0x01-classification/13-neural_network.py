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

    def cost(self, Y, A) -> int:
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
            To prevent division by zero in the logarithm,
        a small value (epsilon) is added,
        such that the expression becomes `1 + epsilon - A` instead of `1 - A`.
            This ensures numerical stability during cost computation.
        (as zero is not defined in log domain)
        """
        m = Y.shape[1]  # number of examples
        epsilon = 1e-7

        Loss_row = (Y * np.log(A)) + ((1-Y) * (np.log((1+epsilon) - A)))
        cost = np.sum(Loss_row) / (-m)
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network’s predictions.

        Parameters
        ----------
        X : np.ndarray of shape (nx, m)
            Input data where:
            - nx is the number of input features.
            - m is the number of examples.

        Y : np.ndarray of shape (1, m)
            Correct labels for the input data.

        Returns
        -------
        tuple
            A tuple containing:
            - predictions : np.ndarray of shape (1, m)
                Binary predictions for each example.
                Labels are 1 if the output activation >= 0.5, else 0.

            - cost : float
                The cost of the network's predictions.
        """

        self.forward_prop(X)  # (1, m) (self.__A2 gets updated)
        cost = self.cost(Y, self.__A2)
        prediction = np.where(self.__A2 >= 0.5, 1, 0)
        return (prediction, cost)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Performs one pass of gradient descent on the neural network.

        Parameters
        ----------
        X : np.ndarray of shape (nx, m)
            Input data where:
            - nx is the number of input features.
            - m is the number of examples.

        Y : np.ndarray of shape (1, m)
            Correct labels for the input data.

        A1 : np.ndarray of shape (nodes, m)
            Activated output from the hidden layer.

        A2 : np.ndarray of shape (1, m)
            Activated output from the output layer (predicted output).

        alpha : float, optional
            Learning rate used in gradient descent (default is 0.05).

        Updates
        -------
        __W1 : np.ndarray
            Weights of the hidden layer, updated in-place.

        __b1 : np.ndarray
            Biases of the hidden layer, updated in-place.

        __W2 : np.ndarray
            Weights of the output layer, updated in-place.

        __b2 : float
            Bias of the output layer, updated in-place.
        """

        m = X.shape[1]
        # calculate the derivatives

# 1. gradient of the loss with respect to the output layer's activations:
        dZ2 = A2 - Y

# 2. gradient with respect to the weights and biases of the second layer:
        dW2 = (dZ2 @ A1.T) / m
        db2 = (np.sum(dZ2, axis=1, keepdims=True))/m

# 4. gradient of the loss with respect to the first layer's activations:
        dZ1 = (self.W2.T @ dZ2) * (A1 * (1-A1))

# 5. gradient with respect to the weights and biases of the first layer:
        dW1 = (dZ1 @ X.T) / m
        db1 = (np.sum(dZ1, axis=1, keepdims=True))/m

# 3. Update the weights and biases of the second layer:
        self.__W2 -= (alpha * dW2)
        self.__b2 -= (alpha * db2)

        # 6. Update the weights and biases of the first layer:
        self.__W1 -= (alpha * dW1)
        self.__b1 -= (alpha * db1)
