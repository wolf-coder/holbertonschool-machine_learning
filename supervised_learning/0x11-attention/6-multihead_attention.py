#!/usr/bin/env python3
"""
MULTIHEAD ATTENTION
"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Class that inherits from tensorflow.keras.layers.Layer
    """

    def __init__(self, dm, h):
        """
        init
        """
        super(MultiHeadAttention, self).__init__()
        self.dm = dm
        self.h = h
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def call(self, Q, K, V, mask):
        """
        call
        """
        batch_size = tf.shape(Q)[0]
        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)
        resh_parm = (batch_size, -1, self.h, self.depth)
        perm = [0, 2, 1, 3]
        q = tf.reshape(q, resh_parm)
        q = tf.transpose(q, perm)
        k = tf.reshape(k, resh_parm)
        k = tf.transpose(k, perm)
        v = tf.reshape(v, resh_parm)
        v = tf.transpose(v, perm)
        output, weights = sdp_attention(q, k, v, mask)
        output = tf.transpose(output, perm)
        output = self.linear(
            tf.reshape(output, (batch_size, -1, self.dm)))
        return output, weights
