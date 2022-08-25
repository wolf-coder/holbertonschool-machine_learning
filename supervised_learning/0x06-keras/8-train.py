#!/usr/bin/env python3
"""
TRAINS MODEL
"""
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
        learning_rate_decay=False,
        alpha=0.1,
        decay_rate=1,
        save_best=False,
        filepath=None,
        verbose=True,
        shuffle=False):
    """
        save best iter
    """
    def scheduler(epoch):
        return alpha / (1 + (decay_rate * epoch))
    callbacks = []
    if early_stopping:
        callbacks.append(K.callbacks.EarlyStopping(patience=patience))
    if learning_rate_decay and validation_data:
        callbacks.append(
            K.callbacks.LearningRateScheduler(
                scheduler, verbose=1))
    if save_best:
        callbacks.append(
            K.callbacks.ModelCheckpoint(
                filepath=filepath,
                save_best_only=True))
    hist_obj = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        callbacks=callbacks,
        validation_data=validation_data,
        shuffle=shuffle)
    return hist_obj
