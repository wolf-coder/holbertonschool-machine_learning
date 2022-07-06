#!/usr/bin/env python3
"""
CONVOLUTION WITH PADDING
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
        convolution on grayscale images with padding
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    pad_h, pad_w = padding
    out_h = (h + (pad_h * 2)) - kh + 1
    out_w = (w + (pad_w * 2)) - kw + 1
    convolved_image = np.zeros(shape=(m, out_h, out_w))
    padded = np.pad(
        array=images, pad_width=(
            (0,), (pad_h,), (pad_w,)), mode="constant", constant_values=0)
    for row in range(0, out_h):
        for col in range(0, out_w):
            convolved_image[:, row, col] = (
                padded[:, row:kh + row, col:kw + col] * kernel).sum(
                    axis=(1, 2))
    return convolved_image
