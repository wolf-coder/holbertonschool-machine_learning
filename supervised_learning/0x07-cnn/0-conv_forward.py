#!/usr/bin/env python3
"""
CONVOLUTIONAL FORWARD PROP
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
        forward propagation over a convolutional layer of a neural network
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    if padding == 'same':
        padding_h = (((h_prev - 1) * sh + kh - h_prev) // 2)
        padding_w = (((w_prev - 1) * sw + kw - w_prev) // 2)
    else:
        padding_h, padding_w = 0, 0
    output_h = ((h_prev - kh + 2 * padding_h) // sh) + 1
    output_w = ((w_prev - kw + 2 * padding_w) // sw) + 1
    padded = np.pad(A_prev, ((0,), (padding_h,),
                    (padding_w,), (0,)), 'constant')
    convolved_image = np.zeros((m, output_h, output_w, c_new))
    for row in range(0, output_h):
        for col in range(0, output_w):
            for kernel in range(c_new):
                convolved_image[:, row, col, kernel] = activation(((padded[
                    :, row * sh:kh + row * sh, col * sw:kw + col * sw, :] * W[
                        :, :, :, kernel]).sum(axis=(1, 2, 3))) + b[
                            :, :, :, kernel])
    return convolved_image
