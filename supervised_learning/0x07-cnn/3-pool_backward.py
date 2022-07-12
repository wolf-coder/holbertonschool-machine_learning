#!/usr/bin/env python3
"""POOLING BACK PROP"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
        back propagation over a pooling layer
    """
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    dA_prev = np.zeros_like(A_prev)

    for image in range(m):
        for row in range(0, h_new):
            for col in range(0, w_new):
                for kernel in range(c_new):
                    sA_prev = A_prev[image, row * sh:kh +
                                     row * sh, col * sw:kw + col * sw, kernel]
                    sdA_prev = dA_prev[image, row * sh:kh +
                                       row * sh, col * sw:kw + col*sw, kernel]

                    if mode == 'max':
                        mask = (sA_prev == np.max(sA_prev))
                        sdA_prev += np.multiply(dA[image,
                                                row, col, kernel], mask)

                    if mode == 'avg':
                        sdA_prev += dA[image, row, col, kernel] / (kh * kw)
    return dA_prev
