#!/usr/bin/env python3
"""identity_block"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """builds an identity block as described in Deep
    Residual Learning for Image Recognition (2015)"""
    hn = K.initializers.he_normal()
    conv1 = K.layers.Conv2D(filters[0], (1, 1),
                            padding='same',
                            kernel_initializer=hn)(A_prev)
    b_norm1 = K.layers.BatchNormalization()(conv1)
    act = K.layers.Activation('relu')(b_norm1)
    conv2 = K.layers.Conv2D(filters[1], (3, 3),
                            padding='same',
                            kernel_initializer=hn)(act)
    b_norm2 = K.layers.BatchNormalization()(conv2)
    act2 = K.layers.Activation('relu')(b_norm2)
    conv3 = K.layers.Conv2D(filters[2], (1, 1),
                            padding='same',
                            kernel_initializer=hn)(act2)
    b_norm3 = K.layers.BatchNormalization()(conv3)
    output = K.layers.Add()([b_norm3, A_prev])
    act3 = K.layers.Activation('relu')(output)
    return act3
