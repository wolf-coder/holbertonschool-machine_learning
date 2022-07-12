#!/usr/bin/env python3
"""
Sensquential
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    builds a neural network with the Keras library
    nx: number of input features to the network
    layers: list containing the number of nodes in each layer
    activations: list containing the activation functions
    lambtha: L2 regularization parameter
    keep_prob: probability that a node will be kept for dropout
    Returns: the keras model
    """
    x = K.Input(shape=(nx,))
    inputs = x
    for i in range(len(layers)):
        if i != 0:
            x = K.layers.Dropout(1 - keep_prob)(x)
        x = K.layers.Dense(
            layers[i],
            activation=(
                activations[i]),
            kernel_regularizer=K.regularizers.L2(lambtha))(x)
    return K.Model(inputs, x)
