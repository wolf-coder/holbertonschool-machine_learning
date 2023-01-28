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
    m, stdv = np.mean(reshped, axis=0), np.std(reshped, axis=0)
    img = reshped.astype('float32')
    img -= np.mean(img)
    img /= np.std(img)
    cov = np.cov(img, rowvar=False)
    lambdas, a = np.linalg.eig(cov)
    delta = np.dot(a, alphas * lambdas)
    pca = (
        (img +
         delta) *
        stdv +
        m).reshape(
        image.shape[0],
        image.shape[1],
        3)
    pca = np.maximum(np.minimum(pca, 255), 0).astype('uint8')
    return pca
