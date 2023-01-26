#!/usr/bin/env python3
"""
Creating a function  that flips an image
"""
import tensorflow as tf


def crop_image(image):
    """image: tf.Tensor"""
    return tf.image.random_crop(image, size)
