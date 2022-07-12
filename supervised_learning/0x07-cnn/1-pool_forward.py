#!/usr/bin/env python3
"""
POOLING FORWARD PROP
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
        forward propagation over a pooling layer
    """
    m, h, w, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    out_h = ((h - kh) // sh) + 1
    out_w = ((w - kw) // sw) + 1
    pooloed_images = np.zeros((m, out_h, out_w, c))
    for row in range(0, out_h):
        for col in range(0, out_w):
            if mode == 'max':
                pooloed_images[:, row, col, :] = (
                    A_prev[:, row * sh:kh + row * sh,
                           col * sw:kw + col * sw, :]).max(axis=(1, 2))
            elif mode == 'avg':
                pooloed_images[:, row, col, :] = np.average(
                    A_prev[:, row * sh:kh + row * sh,
                           col * sw:kw + col * sw, :], axis=(1, 2))
    return pooloed_images
