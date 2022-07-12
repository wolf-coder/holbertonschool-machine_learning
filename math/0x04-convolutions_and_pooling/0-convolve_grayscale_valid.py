#!/usr/bin/env python3
"""
VALID CONVOLUTION
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
        valid convolution on grayscale images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    convolved_image = np.zeros(shape=(m, h - kh + 1, w - kw + 1))
    for row in range(0, h - kh + 1):
        for col in range(0, w - kw + 1):
            convolved_image[:, row, col] = (images[
                :, row:kh + row, col:kw + col] * kernel).sum(axis=(1, 2))
    return convolved_image
