#!/usr/bin/env python3
"""
RNN
 """
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    RNN
    """
    m, h = h_0.shape
    t, m, i = X.shape
    o = rnn_cell.Wy.shape[1]
    hiddn = np.zeros(shape=(t + 1, m, h))
    Y = np.zeros(shape=(t, m, o))
    hiddn[0] = h_0
    for i in range(t):
        hiddn[i + 1, :, :], Y[i, :,
                              :] = rnn_cell.forward(hiddn[i, :, :], X[i, :, :])
    return hiddn, Y
