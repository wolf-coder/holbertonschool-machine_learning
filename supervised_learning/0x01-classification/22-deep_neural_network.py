from functools import cache
import numpy as np

class DeepNeuralNetwork:
    def __init__(self, nx, layers):
        """
        - nx: is the number of input features
        - layers: is a list representing the number of nodes in each layer
        """
        self.__L = len(layers) # L: The number of layers in the neural network.
        self.__cache = {}
        self.__weights = {}

        for layer in range(self.__L):
            # print(index,value)
            Ln = layer + 1
            W_keyName , b_keyName = f'W{Ln}', f'b{Ln}'
            # print(W_keyName)
            if layer == 0:
                prev_n = nx
            else:
                prev_n = layers[layer - 1]

            self.__weights[W_keyName] = np.random.randn(layers[layer], prev_n) * np.sqrt(2/ prev_n)

            self.__weights[b_keyName] = np.zeros ((layers[layer],1))

    @property
    def weights(self):
        return self.__weights

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache 

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        """
        self.__cache["A0"] = X
        for layer in range(1,self.__L+1):
            # A_keyName , b_keyName = f'W{layer + 1}', f'b{layer + 1}'
            W_keyName , b_keyName = f'W{layer}', f'b{layer}'
            A_keyName = f'A{layer}'
            
            # print(W_keyName)
            W = self.__weights[W_keyName]
            b = self.__weights[b_keyName]
            precedand_A = self.__cache[f"A{layer - 1}"]

            Z = W @ precedand_A + b

            A = 1/ (1 + np.exp(-Z)) #
            
            self.__cache[A_keyName] = A
            
        return (A, self.__cache)


    def cost(self, Y, A):
        """
    Calculates the cost of the model using logistic regression
    Y is a numpy.ndarray with shape (1, m) that contains the correct labels for the input data
    A is a numpy.ndarray with shape (1, m) containing the activated output of the neuron for each example
    To avoid division by zero errors, please use 1.0000001 - A instead of 1 - A
    Returns the cost
        """
        safeOne = 1.0000001
        n = A.shape[1]
        loss_row = (Y * np.log(A)) + (1-Y)*np.log(safeOne - A)
        cost = np.sum(loss_row)/ -n
        return cost

    def evaluate(self, X, Y):
        """
    Evaluates the neural network’s predictions
    X is a numpy.ndarray with shape (nx, m) that contains the input data
        nx is the number of input features to the neuron
        m is the number of examples
    Y is a numpy.ndarray with shape (1, m) that contains the correct labels for the input data
    Returns the neuron’s prediction and the cost of the network, respectively
        The prediction should be a numpy.ndarray with shape (1, m) containing the predicted labels for each example
        The label values should be 1 if the output of the network is >= 0.5 and 0 otherwise
        """
        
        # extract the Output from Forward_prop result
        Forward_prop = self.forward_prop(X)
        Output = Forward_prop[0] # (1xm)
        
        predictions = np.where(Output>=0.5,1,0) # (1xm)

        cost = self.cost(Y,Output)

        return (predictions, cost)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
    - Calculates one pass of gradient descent on the neural network.
    - Y is a numpy.ndarray with shape (1, m) that contains the correct labels for the input data.
    - cache is a dictionary containing all the intermediary values of the network.
        - Note that Cache[L] is the output of the neural network
    - alpha is the learning rate.
    - Updates the private attribute __weights.
    - You are allowed to use one loop.
        """
        m = Y.shape[1]

        for layer in range(self.L, 0, -1):
            A_current = cache[f"A{layer}"]
            A_precedant = cache[f"A{layer -1}"]
            if layer == self.__L:
                dz = (A_current - Y)
            else:
                dz = dA_precedent * (A_current * (1 - A_current))

            dW = (dz @ A_precedant.T) / m
            db = (np.sum(dz, axis=1, keepdims=True)) / m

            W = self.weights[f"W{layer}"]
            dA_precedent = W.T @ dz

            self.__weights[f"W{layer}"] = (
                self.__weights[f"W{layer}"] - (alpha * dW))
            self.__weights[f"b{layer}"] = (
                self.__weights[f"b{layer}"] - (alpha * db))


    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        - Trains the deep neural network
        - X is a numpy.ndarray with shape (nx, m) that contains the input data
            + nx is the number of input features to the neuron
            + m is the number of examples
        - Y is a numpy.ndarray with shape (1, m) that contains the correct labels for the input data
        - iterations is the number of iterations to train over
            + if iterations is not an integer, raise a TypeError with the exception iterations must be an integer
            + if iterations is not positive, raise a ValueError with the exception iterations must be a positive integer
        - alpha is the learning rate
            + if alpha is not a float, raise a TypeError with the exception alpha must be a float
            + if alpha is not positive, raise a ValueError with the exception alpha must be positive
        - All exceptions should be raised in the order listed above
        - Updates the private attributes __weights and __cache
        - You are allowed to use one loop
        - Returns the evaluation of the training data after iterations of training have occurred
        """
        for i in range(iterations):
            self.forward_prop(X) # executed to calculate a forward pass  and to update the self._cache 
            self.gradient_descent(Y, self.__cache,alpha=alpha) # this will calculate a backward pass and updates  the weights
        return self.evaluate(X,Y)
