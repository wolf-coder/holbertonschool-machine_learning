#!/usr/bin/env python3
"""function that hat trains a model using mini-batch gradient descent"""
import tensorflow.keras as K


def train_model(
        network,
        data,
        labels,
        batch_size,
        epochs,
        validation_data=None,
        early_stopping=False,
        patience=0,
        verbose=True,
        shuffle=False):
    """train the model using early stopping"""
    model = network
    callback = None
    if early_stopping:
        callback = K.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience)
    hist_obj = model.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        callbacks=[callback],
        validation_data=validation_data,
        shuffle=shuffle)
    return hist_obj
