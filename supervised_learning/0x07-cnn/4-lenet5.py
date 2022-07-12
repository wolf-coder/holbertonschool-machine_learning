#!/usr/bin/env python3
"""LENET-5"""
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """
        modified version of the LeNet-5 architecture using tensorflow
    """
    kernel = tf.keras.initializers.VarianceScaling(scale=2.0)
    L1 = tf.layers.conv2d(
        inputs=x,
        filters=6,
        kernel_size=(
            5,
            5),
        kernel_initializer=kernel,
        padding="SAME",
        activation='relu')
    L2 = tf.layers.max_pooling2d(
        inputs=L1, pool_size=(
            2, 2), strides=(
            2, 2))
    L3 = tf.layers.conv2d(
        inputs=L2,
        filters=16,
        kernel_size=(
            5,
            5),
        kernel_initializer=kernel,
        activation='relu')
    L4 = tf.layers.max_pooling2d(
        inputs=L3, pool_size=(
            2, 2), strides=(
            2, 2))
    L4 = tf.layers.flatten(L4)
    L5 = tf.layers.dense(
        inputs=L4,
        units=120,
        kernel_initializer=kernel,
        activation='relu')
    L6 = tf.layers.dense(
        inputs=L5,
        units=84,
        kernel_initializer=kernel,
        activation='relu')
    L7 = tf.layers.dense(inputs=L6, units=10, kernel_initializer=kernel,)
    Y_pred = tf.nn.softmax(L7)
    loss = tf.losses.softmax_cross_entropy(y, L7)
    Training = tf.train.AdamOptimizer().minimize(loss)
    accuracy = tf.reduce_mean(
        tf.cast(
            tf.equal(
                tf.argmax(
                    L7, axis=1), tf.argmax(
                    y, axis=1)), "float32"))
    return Y_pred, Training, loss, accuracy
