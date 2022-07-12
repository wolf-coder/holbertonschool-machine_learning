#!/usr/bin/env python3
"""
TEST N.N
"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
        TEST NEURAL NETWORK
    """
    return network.evaluate(data, labels, verbose=verbose)
