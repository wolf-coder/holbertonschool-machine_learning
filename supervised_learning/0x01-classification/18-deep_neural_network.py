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
