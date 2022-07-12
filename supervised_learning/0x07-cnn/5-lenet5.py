#!/usr/bin/env python3
"""LENET-5"""
import tensorflow.keras as K


def lenet5(X):
    """
        modified version of the LeNet-5 architecture using Keras
    """
    kernel = K.initializers.he_normal()
    L1 = K.layers.Conv2D(6, (5, 5), kernel_initializer=kernel,
                         padding='SAME', activation='relu')(X)
    L2 = K.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(L1)
    L3 = K.layers.Conv2D(
        16, (5, 5), kernel_initializer=kernel, activation='relu')(L2)
    L4 = K.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(L3)
    L4_2 = K.layers.Flatten()(L4)
    L5 = K.layers.Dense(
        120,
        kernel_initializer=kernel,
        activation='relu')(L4_2)
    L6 = K.layers.Dense(84, kernel_initializer=kernel, activation='relu')(L5)
    L7 = K.layers.Dense(
        10,
        activation='softmax',
        kernel_initializer=kernel)(L6)
    model = K.models.Model(inputs=X, outputs=L7)
    model.compile(
        optimizer=K.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model
