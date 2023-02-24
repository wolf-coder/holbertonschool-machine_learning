#!/usr/bin/env python3
"""
Attention
"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    Inherits from tensorflow.keras.layers.Layer
    """
    def __init__(self, units):
        """
        Init
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units=units)
        self.U = tf.keras.layers.Dense(units=units)
        self.V = tf.keras.layers.Dense(units=1)

    def call(self, s_prev, hidden_states):
        """
        Call
        """
        s_prev = tf.expand_dims(s_prev, 1)
        o = self.V(tf.nn.tanh(self.W(s_prev) + self.U(
            hidden_states)))
        w = tf.nn.softmax(o, axis=1)
        context = tf.reduce_sum(w * hidden_states, axis=1)
        return context, w
