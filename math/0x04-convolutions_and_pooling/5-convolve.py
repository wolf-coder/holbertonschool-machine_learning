#!/usr/bin/env python3
"""
MULTIPLE KERNELS
"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
        convolution on images using multiple kernels
    """
    m, h, w, c = images.shape
    kh, kw, c, nc = kernels.shape
    sh, sw = stride
    if padding == 'valid':
        padding_h, padding_w = 0, 0
    elif padding == 'same':
        padding_h = (((h - 1) * sh + kh - h) // 2) + 1
        padding_w = (((w - 1) * sw + kw - w) // 2) + 1
    else:
        padding_h, padding_w = padding
    out_h = ((h - kh + 2 * padding_h) // sh) + 1
    out_w = ((w - kw + 2 * padding_w) // sw) + 1
    padded = np.pad(images, ((0,), (padding_h,),
                    (padding_w,), (0,)), 'constant')
    convolved_image = np.zeros(shape=(m, out_h, out_w, nc))
    for row in range(0, out_h):
        for col in range(0, out_w):
            for ker in range(nc):
                convolved_image[:,
                                row,
                                col,
                                ker] = (padded[:,
                                               row * sh:kh + row * sh,
                                               col * sw:kw + col * sw,
                                               :] * kernels[:,
                                                            :,
                                                            :,
                                                            ker]).sum(axis=(1,
                                                                            2,
                                                                            3))
    return convolved_image
