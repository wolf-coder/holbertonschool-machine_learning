#!/usr/bin/env python3
"""
Creating a function  that shears an image
"""
import tensorflow as tf


def shear_image(image, intensity):
    """image: tf.Tensor"""
    return tf.keras.preprocessing.image.random_shear(
        image.numpy(), intensity, channel_axis=2)
