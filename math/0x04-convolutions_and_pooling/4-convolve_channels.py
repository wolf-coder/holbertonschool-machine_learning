#!/usr/bin/env python3
"""
CONVOLUTION WITH CHANNELS
"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
        Convolution on images with Channels
    """
    m, h, w, c = images.shape
    kh, kw, c = kernel.shape
    sh, sw = stride
    if padding == 'valid':
        pad_h, pad_w = 0, 0
    elif padding == 'same':
        pad_h = (((h - 1) * sh + kh - h) // 2) + 1
        pad_w = (((w - 1) * sw + kw - w) // 2) + 1
    else:
        pad_h, pad_w = padding
    out_h = ((h - kh + 2 * pad_h) // sh) + 1
    out_w = ((w - kw + 2 * pad_w) // sw) + 1
    padded_image = np.pad(images, ((0,), (pad_h,), (pad_w,), (0,)), 'constant')
    convolved_image = np.zeros(shape=(m, out_h, out_w))
    for row in range(0, out_h):
        for col in range(0, out_w):
            convolved_image[:, row, col] = (
                    padded_image[:, row * sh:kh + row * sh, col * sw:kw + col * sw, :]
                    * kernel).sum(axis=(1, 2, 3))
    return convolved_image
