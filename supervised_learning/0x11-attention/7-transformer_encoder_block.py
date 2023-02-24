#!/usr/bin/env python3
"""
ENCODER BLOCK
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """
    transformer encoder block
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        init
        """
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden,
                                                  activation="relu")
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
        call
        """
        out, _ = self.mha(x, x, x, mask)
        out = self.dropout1(out, training=training)
        y = self.layernorm1(x + out)
        out = self.dense_hidden(y)
        out = self.dense_output(out)
        out = self.dropout2(out, training=training)
        seq_out = self.layernorm2(y + out)
        return seq_out
