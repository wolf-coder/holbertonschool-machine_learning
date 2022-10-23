#!/usr/bin/env python3
"""
bidirectional RNN
"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    bidirectional RNN
    """
    t, m, _ = X.shape
    frwd = []
    bkwrd = []
    prev = h_0
    hnext = h_t
    for i in range(t):
        frwd.append(bi_cell.forward(prev, X[i]))
        prev = frwd[i]
        bkwrd.append(bi_cell.backward(hnext, X[t-i-1]))
        hnext = bkwrd[i]
    bkwrd.reverse()
    H = np.concatenate((np.array(frwd), np.array(bkwrd)), axis=2)
    y = bi_cell.output(H)
    return H, y
