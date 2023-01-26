#!/usr/bin/env python3
"""
Creating a function  that changes the hue of an image
"""
import tensorflow as tf


def change_hue(image, delta):
    """image: tf.Tensor"""
    return tf.image.adjust_hue(image, delta)
