#!/usr/bin/env python3
"""
Neuron
"""
import numpy as np
import matplotlib.pyplot as plt


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
        activation_output = 1 / (1 + np.exp(-z))
        self.__A = activation_output
        return self.__A

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
        epsilon = 1e-7: a standard practice for avoiding log(0) while
        keeping output close to expected.
        """
        epsilon = 1e-7
        m = Y.shape[1]
        loss_row = Y * np.log(A) + (1 - Y) * np.log((1 + epsilon) - A)
        cost = -np.sum(loss_row) / m
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions.

        Parameters
        ----------
        X : np.ndarray of shape (nx, m)
            Input data, where:
            - nx is the number of input features
            - m is the number of examples

        Y : np.ndarray of shape (1, m)
            Correct labels for the input data.

        Returns
        -------
        tuple
            A tuple containing:
            - predictions: np.ndarray of shape (1, m)
                Binary predictions (0 or 1) for each example.
            - cost: float
                The cost of the model using logistic regression.
        """
        activated_outputs = self.forward_prop(X)
        predictions = np.where(activated_outputs >= 0.5, 1, 0)
        return predictions, self.cost(Y, activated_outputs)

    def gradient_descent(self, X, Y, A, alpha=0.05) -> None:
        """
        Performs one pass of gradient descent on the neuron.

        Parameters
        ----------
        X : np.ndarray of shape (nx, m)
            Input data:
            - nx: number of input features
            - m: number of examples

        Y : np.ndarray of shape (1, m)
            Correct labels for the input data.

        A : np.ndarray of shape (1, m)
            Activated output of the neuron for each example.

        alpha : float, optional
            Learning rate (default is 0.05).

        Updates
        -------
        self.__W : np.ndarray
            Weight vector, updated in-place using gradient descent.

        self.__b : float
            Bias, updated in-place.
         """
        m = Y.shape[1]

        # Compute the gradients
        dz = A - Y                           # shape: (1, m)
        dw = np.dot(X, dz.T) / m            # shape: (nx, 1)
        db = np.sum(dz) / m                 # scalar

        # Update parameters
        self.__W -= alpha * dw.T            # self.__W: shape (1, nx)
        self.__b -= alpha * db              # self.__b: scalar


    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
        Trains the neuron by updating the private attributes __W, __b, and __A

        Parameters
        ----------
        X : np.ndarray of shape (nx, m)
            Input data.

        Y : np.ndarray of shape (1, m)
            Correct labels.

        iterations : int
            Number of training iterations.

        alpha : float
            Learning rate.

        verbose : bool
            If True, prints cost every `step` iterations.

        graph : bool
            If True, plots the cost over iterations.

        step : int
            Interval of iterations to print/plot.

        Returns
        -------
        tuple
            The evaluation (predictions, cost) after training.

        Raises
        ------
        TypeError
            If iterations is not an int, or alpha not a float, or step not an int (when needed).

        ValueError
            If iterations or alpha or step are invalid values.
        """

        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        steps = []

        for i in range(iterations + 1):
            A = self.forward_prop(X)
            if i % step == 0 or i == iterations:
                cost = self.cost(Y, A)
                costs.append(cost)
                steps.append(i)
                if verbose:
                    print(f"Cost after {i} iterations: {cost}")
            if i < iterations:
                self.gradient_descent(X, Y, A, alpha)

        if graph:
            plt.plot(steps, costs, 'b')
            plt.xlabel("Iteration")
            plt.ylabel("Cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)
