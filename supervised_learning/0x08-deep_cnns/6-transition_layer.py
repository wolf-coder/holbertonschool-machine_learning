#!/usr/bin/env python3
"""Dense Block module"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """  builds a transition layer as described in Densely
    Connected Convolutional Networks"""
    x = K.layers.BatchNormalization()(X)
    x = K.layers.Activation('relu')(x)
    factor = int(nb_filters * compression)
    x = K.layers.Conv2D(factor, (1, 1),
                        kernel_initializer='he_normal',
                        padding='same')(x)
    x = K.layers.AveragePooling2D((2, 2), strides=(2, 2))(x)
    return x, factor
