#!/usr/bin/env python3
"""
SAVE & LOAD WEIGHTS
"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """
        SAVE WEIGHTS
    """
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """
        LOAD WEIGHTS
    """
    return network.load_weights(filename)
