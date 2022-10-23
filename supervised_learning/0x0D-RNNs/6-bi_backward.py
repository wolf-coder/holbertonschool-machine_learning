#!/usr/bin/env python3
"""
bidirectional RNN
"""
import numpy as np


class BidirectionalCell:
    """
    bidirectional RNN
    """

    def __init__(self, i, h, o):
        """
        init
        """
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))
        self.Whf = np.random.normal(0.0, 1.0, (i + h, h))
        self.Whb = np.random.normal(0.0, 1.0, (i + h, h))
        self.Wy = np.random.normal(0.0, 1.0, (2 * h, o))

    def forward(self, h_prev, x_t):
        """
        forward
        """
        x = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(x, self.Whf) + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """
        backward
        """
        x =  np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(np.matmul(x, self.Whb) + self.bhb)
        return h_prev
