#!/usr/bin/env python3
"""
Encoder for RNN 
"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    Inherits from tensorflow.keras.layers.Layer
    """
    def __init__(self, vocab, embedding, units, batch):
        """
        Constructor
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def initialize_hidden_state(self):
        """
        Initialize
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """
        Call
        """
        x = self.embedding(x)
        return self.gru(x, initial)