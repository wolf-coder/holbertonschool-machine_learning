#!/usr/bin/env python3
"""
Creating a function  that flips an image
"""
import tensorflow as tf


def rotate_image(image):
    """image: tf.Tensor"""
    return tf.image.rot90(image)
