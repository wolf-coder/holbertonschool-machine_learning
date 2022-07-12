#!/usr/bin/env python3
"""LENET-5"""
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """
        modified version of the LeNet-5 architecture using tensorflow
    """
    kernel = tf.contrib.layers.variance_scaling_initializer()
    L1_conv = tf.layers.conv2d(x, 6, (5, 5), kernel, "SAME", 'relu')
    L2_pool = tf.layers.max_pooling2d(L1_conv, (2, 2), (2, 2))
    L3_conv = tf.layers.conv2d(L2_pool, 16, (5, 5), kernel, 'relu')
    L4_pool = tf.layers.max_pooling2d(L3_conv, (2, 2), (2, 2))
    L4_flat = tf.contrib.layers.flatten(L4_pool)
    L5_fc = tf.layers.dense(L4_flat, 120, kernel, 'relu')
    L6_fc = tf.layers.dense(L5_fc, 84, kernel, 'relu')
    L7_fc = tf.layers.dense(L6_fc, 10, kernel)
    Y_pred = tf.nn.softmax(L7_fc)
    loss = tf.losses.softmax_cross_entropy(y, L7_fc)
    Train_op = tf.train.AdamOptimizer().minimize(loss)
    acc = tf.reduce_mean(
        tf.cast(
            tf.equal(
                tf.argmax(
                    L7_fc, axis=1), tf.argmax(
                    y, axis=1)), "float32"))
    return Y_pred, Train_op, loss, acc
