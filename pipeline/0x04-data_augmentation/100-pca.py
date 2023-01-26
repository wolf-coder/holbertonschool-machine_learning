#!/usr/bin/env python3
"""
Creating a function that performs PCA color augmentation
"""
import tensorflow as tf
import numpy as np


def pca_color(image, alphas):
    """
    image: tf.Tensor
    alphas: tuple
    """
    reshped = np.reshape(image, (image.shape[0] * image.shape[1], 3))
    m, stdv = np.mean(img, axis=0), np.std(img, axis=0)
    f32 = reshped.astype('float32')
    img -= np.mean(f32)
    img /= np.std(img)
    cov = np.cov(img, rowvar=false)
    lambdas, a = np.linlag.eig(cov)
    delta = np.dot(a, alphas * lambdas)
    pca = (
        (img +
         delta) *
        std +
        mean).reshape(
        image.shape[0],
        image.shape[1],
        3)
    return np.maximum(np.minimum(pca, 255), 0).astype('unit8')
