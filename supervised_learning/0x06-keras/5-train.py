#!/usr/bin/env python3
"""
    Trains a model
"""
import tensorflow.keras as K


def train_model(
        network,
        data,
        labels,
        batch_size,
        epochs,
        validation_data=None,
        verbose=True,
        shuffle=False):
    """
    trains a model
    """
    model = network
    hist_obj = model.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        validation_data=validation_data,
        shuffle=shuffle)
    return hist_obj
