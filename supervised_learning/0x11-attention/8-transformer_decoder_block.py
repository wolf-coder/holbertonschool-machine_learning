#!/usr/bin/env python3
"""
DECODER BLOCK
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """
    transformer decoder block
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        init
        """
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation="relu")
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        call
        """
        out_mha_1, _ = self.mha1(x, x, x, look_ahead_mask)
        out_mha_1 = self.dropout1(out_mha_1, training=training)
        out_nrm_1 = self.layernorm1(x + out_mha_1)
        out_mha_2, _ = self.mha2(
            out_nrm_1, encoder_output, encoder_output, padding_mask)
        out_mha_2 = self.dropout2(out_mha_2, training=training)
        out_nrm_2 = self.layernorm2(out_mha_2 + out_nrm_1)
        output = self.dense_hidden(out_nrm_2)
        output = self.dense_output(output)
        output = self.dropout3(output, training=training)
        out_nrm_3 = self.layernorm3(output + out_nrm_2)
        return out_nrm_3
