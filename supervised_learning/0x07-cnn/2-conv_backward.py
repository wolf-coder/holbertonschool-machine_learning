#!/usr/bin/env python3
"""CONVOLUTIONAL BACK PROP"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
        back propagation over a convolutional layer of a neural network
    """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    if padding == 'valid':
        padding_h, padding_w = 0, 0
    else:
        padding_h = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        padding_w = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    A_prev_padded = np.pad(
        A_prev, ((0,), (padding_h,), (padding_w,), (0,)), 'constant')

    dW = np.zeros(shape=(kh, kw, c_prev, c_new))
    dA_prev = np.zeros_like(A_prev_padded)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    for image in range(m):
        for row in range(0, h_new):
            for col in range(0, w_new):
                for kernel in range(c_new):
                    dW[:,
                       :,
                       :,
                       kernel] += A_prev_padded[image,
                                                row * sh:kh + row * sh,
                                                col * sw:kw + col * sw,
                                                :] * dZ[image,
                                                        row,
                                                        col,
                                                        kernel]
                    dA_prev[image,
                            row * sh:kh + row * sh,
                            col * sw:kw + col * sw,
                            :] += dZ[image,
                                     row,
                                     col,
                                     kernel] * W[:,
                                                 :,
                                                 :,
                                                 kernel]
    dA_prev = dA_prev[:, padding_h:h_prev +
                      padding_h, padding_w:padding_w + w_prev, :]
    return dA_prev, dW, db
