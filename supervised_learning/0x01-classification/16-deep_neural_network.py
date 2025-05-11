import numpy as np

class DeepNeuralNetwork:
    def __init__(self, nx, layers):
        """
        - nx: is the number of input features
        - layers: is a list representing the number of nodes in each layer
        """
        self.L = len(layers) # L: The number of layers in the neural network.
        self.cache = {}
        self.weights = {}

        for index in range(self.L):
            # print(index,value)
            Ln = index + 1
            W_keyName , b_keyName = f'W{Ln}', f'b{Ln}'
            # print(W_keyName)
            if index == 0:
                prev_n = nx
            else:
                prev_n = layers[index - 1]

            self.weights[W_keyName] = np.random.randn(layers[index], prev_n) * np.sqrt(2/ prev_n)

            self.weights[b_keyName] = np.zeros ((layers[index],1))
