#!/usr/bin/env python3
"""
Creating a function  that changes the brightness of an image
"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """image: tf.Tensor"""
    return tf.image.adjust_brightness(image, max_delta)
