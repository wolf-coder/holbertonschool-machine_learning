#!/usr/bin/env python3
"""
Decoder
"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    Class that inherits from tensorflow.keras.layers.Layer
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        init
        """
        super(RNNDecoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """
        Call
        """
        attention = SelfAttention(self.units)
        context, _ = attention(s_prev, hidden_states)
        x = self.embedding(x)
        concat_x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)
        y, hidden = self.gru(concat_x)
        output = self.F(tf.reshape(y, (-1, y.shape[2])))
        return output, hidden
