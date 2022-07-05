#!/usr/bin/env python3
"""
SAVE & LOAD CONFIG
"""
import tensorflow.keras as K


def save_config(network, filename):
    """
        SAVE CONFIG
    """
    with open(filename, "w") as f:
        f.write(network.to_json())


def load_config(filename):
    """
        LOAD CONFIG
    """
    with open(filename, "r") as f:
        conf = f.read()
    return K.models.model_from_json(conf)
