#!/usr/bin/env python3
"""
POOLING
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
        Pooling on images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride
    output_h = ((h - kh) // sh) + 1
    output_w = ((w - kw) // sw) + 1
    pooled_images = np.zeros(shape=(m, output_h, output_w, c))
    for row in range(0, output_h):
        for col in range(0, output_w):
            if mode == 'max':
                pooled_images[:, row, col, :] = (
                    images[:, row * sh:kh + row * sh,
                           col * sw:kw + col * sw, :]).max(axis=(1, 2))
            elif mode == 'avg':
                pooled_images[:, row, col, :] = np.average(
                    images[:, row * sh:kh + row * sh,
                           col * sw:kw + col * sw, :], axis=(1, 2))
    return pooled_images
