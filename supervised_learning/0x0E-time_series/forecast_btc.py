#!/usr/bin/env python3
"""
Creates, trains, and validates a
keras model for the forecasting of BTC
"""
import tensorflow as tf
from preprocess_data import preprocess
import matplotlib.pyplot as plt
import pandas as pd

model = tf.keras.models.Sequential(
    [tf.keras.layers.LSTM(32, return_sequences=False),
     tf.keras.layers.Dense(units=1)])

callbacks = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=2, mode='min')
model.compile(loss=tf.losses.MeanSquaredError(),
              optimizer=tf.optimizers.Adam(),
              metrics=[tf.metrics.MeanAbsoluteError()])

window = preprocess()
history = model.fit(window.train, epochs=20,
                    validation_data=window.val,
                    callbacks=[callbacks])

val_performance = model.evaluate(window.val)
performance = model.evaluate(window.test, verbose=0)

model.save('BTC_lstm.h5')
print(val_performance)
print(performance)
