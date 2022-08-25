#!/usr/bin/env python3
""" Dense Block module"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """builds a dense block as described in
    Densely Connected Convolutional Networks"""
    prev_output = X

    init = K.initializers.he_normal()
    for _ in range(layers):
        b_norm1 = K.layers.BatchNormalization()(prev_output)
        activation1 = K.layers.Activation('relu')(b_norm1)
        conv1 = K.layers.Conv2D(4 * growth_rate, (1, 1),
                                kernel_initializer=init,
                                padding='same')(activation1)
        b_norm2 = K.layers.BatchNormalization()(conv1)
        activation2 = K.layers.Activation('relu')(b_norm2)
        conv2 = K.layers.Conv2D(growth_rate, (3, 3),
                                kernel_initializer=init,
                                padding='same')(activation2)
        nb_filters += growth_rate
        prev_output = K.layers.Concatenate()([prev_output, conv2])
    return prev_output, nb_filters
