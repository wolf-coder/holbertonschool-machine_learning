#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pickle

class DeepNeuralNetwork:
    def __init__(self, nx, layers, activation='sig'):
        """
        - nx: is the number of input features
        - layers: is a list representing the number of nodes in each layer
        """
        self.__L = len(layers) # L: The number of layers in the neural network.
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

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
    def activation(self):
        return self.__weights
    

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


            Z = W @ precedand_A + b # (10, m)
            
            # Use sigmoid for hidden layers
            if layer < self.__L:
                if self.activation == 'sig':
                    A = 1 / (1 + np.exp(-Z))
                else: #activation is tanh
                    A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
            else:
                # Use softmax for the output layer
                exp_Z = np.exp(Z)  # For numerical stability
                A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
                
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
        m = Y.shape[1] # number of examples
        loss_row = -( Y * np.log(A)) # (C, m)
        cost = np.sum(loss_row / m)
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
        
        # get the Output from Forward_prop result
        Output, _ = self.forward_prop(X) # Output: (C x m)

        # using mecanism of the method `one_hot_encode`
        predictions = np.argmax(Output,axis=0) # (m )
        A_one_hot = np.zeros((10, predictions.shape[0]))
        A_one_hot[predictions, np.arange(predictions.shape[0])] = 1

        
        cost = self.cost(Y,Output)
        
        return (A_one_hot, cost)

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
                if self.activation == 'sig':
                    dz = dA_precedent * (A_current * (1 - A_current))
                elif self.activation == 'tanh':
                    dz = dA_precedent * (1 - np.power(A_current,2))  # tanh der
                    

            dW = (dz @ A_precedant.T) / m
            db = (np.sum(dz, axis=1, keepdims=True)) / m

            W = self.weights[f"W{layer}"]
            dA_precedent = W.T @ dz

            self.__weights[f"W{layer}"] = (
                self.__weights[f"W{layer}"] - (alpha * dW))
            self.__weights[f"b{layer}"] = (
                self.__weights[f"b{layer}"] - (alpha * db))


    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
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
        cost_points = []
        for i in range(iterations+1):
            Output, cache = self.forward_prop(X) # executed to calculate a forward pass  and to update the self._cache 
            self.gradient_descent(Y, self.__cache,alpha=alpha) # this will calculate a backward pass and updates  the weights
            cost_points.append(self.cost(Y, Output))
            if verbose is True and i % step == 0:
                print("Cost after {} iterations: {}"
                      .format(i, cost_points[i]))
        if graph is True:
            plt.plot(np.arange(0, iterations + 1), cost_points)
            plt.title("Training cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.show()

        return self.evaluate(X,Y)


    def save(self, filename):
        """
        -  Saves the instance object to a file in pickle format:
        * Filename is the file to which the object should be saved
        * If filename does not have the extension .pkl, add it
        """
        if not filename.endswith(".pkl"):
            filename += ".pkl"  
        with open(filename, 'wb') as File:
            pickle.dump(self, File)

    @staticmethod
    def load(filename):
        """Load function to load a pickled object"""
        try:
            with open(filename, 'rb') as File:
                b = pickle.load(File)
                return b
        except (OSError, IOError) as e:
            return None
