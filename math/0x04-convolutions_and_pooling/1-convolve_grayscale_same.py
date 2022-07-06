#!/usr/bin/env python3
"""
SAME COMVOLUTION
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
        Same convolution on grayscale images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    convolved_image = np.zeros(shape=(m, h, w))
    if kh % 2:
        padding_h = (kh - 1) // 2
    else:
        padding_h = kh // 2
    if kw % 2:
        padding_w = (kw - 1) // 2
    else:
        padding_w = kw // 2
    padded_image = np.pad(array=images, pad_width=(
        (0,), (padding_h,), (padding_w,)), mode="constant", constant_values=0)
    for row in range(0, h):
        for col in range(0, w):
            convolved_image[:, row, col] = (padded_image[
                :, row:kh + row, col:kw + col] * kernel).sum(axis=(1, 2))
    return convolved_image
