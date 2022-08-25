#!/usr/bin/env python3
"""
PREDICT USING N.N
"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
        PREDICT USING NEURAL NETWORK
    """
    return network.predict(data, verbose=verbose)
