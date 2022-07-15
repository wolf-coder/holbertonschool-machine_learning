#!/usr/bin/env python3
"""YOLO module"""
import numpy as np
import tensorflow as tf


class NST:
    """NST class"""
    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """init function"""
        if (not isinstance(style_image, np.ndarray) or len(
                style_image.shape) != 3 or style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if (not isinstance(content_image, np.ndarray) or len(
                content_image.shape) != 3 or content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        tf.enable_eager_execution()
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()

    @staticmethod
    def scale_image(image):
        """
            rescales an image
        """
        if (not isinstance(image, np.ndarray) or len(
                image.shape) != 3 or image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        h, w, _ = image.shape
        h_new = w_new = 512
        if h < w:
            h_new = int(h * 512 / w)
        elif h > w:
            w_new = int(w * 512 / h)
        image = tf.expand_dims(image, 0)
        image = tf.image.resize(image, (h_new, w_new))
        image = image / 255
        return tf.clip_by_value(image, clip_value_min=0, clip_value_max=1)

    def load_model(self):
        """creates model"""
        base_model = tf.keras.applications.VGG16(include_top=False)
        base_model.trainable = False
        base_model.save("base_model")
        model = tf.keras.models.load_model(
            "base_model", {
                'MaxPooling2D': tf.keras.layers.AveragePooling2D})
        style_outputs = [model.get_layer(layer).output for layer in self.style_layers]
        content_output = [model.get_layer(self.content_layer).output]
        self.model = tf.keras.models.Model(
            model.input, style_outputs + content_output)
