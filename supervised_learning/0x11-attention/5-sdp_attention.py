#!/usr/bin/env python3
"""
SCALED DOT PRODUCT
"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    scaled dot product
    """
    sqr = tf.math.sqrt(
        tf.cast(tf.shape(K)[-1], tf.float32))
    scld = tf.matmul(Q, K, transpose_b=True) / sqr
    if mask:
        scld += (mask * -1e9)
    sfmx = tf.nn.softmax(scld, axis=-1)
    return tf.matmul(sfmx, V), sfmx
