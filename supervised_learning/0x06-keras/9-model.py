#!/usr/bin/env python3
"""SAVE & LOAD"""
import tensorflow.keras as K


def save_model(network, filename):
    """
        SAVE
    """
    network.save(filename)


def load_model(filename):
    """
        LOAD
    """
    return K.models.load_model(filename)
