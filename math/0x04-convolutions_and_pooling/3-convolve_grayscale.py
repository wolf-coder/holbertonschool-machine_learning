#!/usr/bin/env python3
"""
STRIDED GRAYSCALE
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
        convolution on grayscale images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride
    if padding == 'valid':
        padding_h, padding_w = 0, 0
    elif padding == 'same':
        padding_h = (((h - 1) * sh + kh - h) // 2) + 1
        padding_w = (((w - 1) * sw + kw - w) // 2) + 1
    else:
        padding_h, padding_w = padding
    output_h = ((h - kh + 2 * padding_h) // sh) + 1
    output_w = ((w - kw + 2 * padding_w) // sw) + 1
    padded = np.pad(images, ((0,), (padding_h,), (padding_w,)), 'constant')
    convolved_image = np.zeros(shape=(m, output_h, output_w))
    for row in range(0, output_h):
        for col in range(0, output_w):
            convolved_image[:, row, col] = (
                    padded[:, row * sh:kh + row * sh, col * sw:kw + col * sw]
                    * kernel).sum(axis=(1, 2))
    return convolved_image
