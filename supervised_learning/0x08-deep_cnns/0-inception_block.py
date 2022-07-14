#!/usr/bin/env python3
"""INCEPTION BLOCK"""

import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
        builds an inception block as described in Going
        Deeper with Convolutions
    """
    F1 = K.layers.Conv2D(filters[0], (1, 1),
                         padding="same",
                         activation="relu")(A_prev)
    F3R = K.layers.Conv2D(filters[1], (1, 1),
                          padding="same",
                          activation="relu")(A_prev)
    F3 = K.layers.Conv2D(filters[2], (3, 3),
                         padding="same",
                         activation="relu")(F3R)
    F5R = K.layers.Conv2D(filters[3], (1, 1),
                          padding="same",
                          activation="relu")(A_prev)
    F5 = K.layers.Conv2D(filters[4], (5, 5),
                         padding="same",
                         activation="relu")(F5R)
    pool = K.layers.MaxPooling2D(pool_size=(3, 3),
                                 strides=(1, 1),
                                 padding="same")(A_prev)
    FPP = K.layers.Conv2D(filters[5], (1, 1),
                          padding="same",
                          activation="relu")(pool)
    return K.layers.Concatenate(axis=-1)([F1, F3, F5, FPP])
