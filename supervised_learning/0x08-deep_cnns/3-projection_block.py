#!/usr/bin/env python3
"""projection block"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """ projection block function"""
    hn = K.initializers.he_normal()
    conv1 = K.layers.Conv2D(filters[0], (1, 1),
                            strides=s, padding='same',
                            kernel_initializer=hn)(A_prev)
    b_norm1 = K.layers.BatchNormalization()(conv1)
    act1 = K.layers.Activation('relu')(b_norm1)
    conv2 = K.layers.Conv2D(filters[1], (3, 3), padding='same',
                            kernel_initializer=hn)(act1)
    b_norm2 = K.layers.BatchNormalization()(conv2)
    act2 = K.layers.Activation('relu')(b_norm2)
    conv3 = K.layers.Conv2D(filters[2], (1, 1), padding='same',
                            kernel_initializer=hn)(act2)
    b_norm3 = K.layers.BatchNormalization()(conv3)
    conv4 = K.layers.Conv2D(filters[2], (1, 1),
                            strides=s, padding='same',
                            kernel_initializer=hn)(A_prev)
    b_norm4 = K.layers.BatchNormalization()(conv4)
    output = K.layers.Add()([b_norm3, b_norm4])
    output = K.layers.Activation('relu')(output)
    return output
