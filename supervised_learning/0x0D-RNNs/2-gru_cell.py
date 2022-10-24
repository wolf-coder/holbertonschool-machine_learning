#!/usr/bin/env python3
"""
GRU CLASS
"""
import numpy as np


class GRUCell:
    """
    GRU CLASS
    """

    def __init__(self, i, h, o):
        """ i: dimensionality of the data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs """
        self.Wz = np.random.normal(0.0, 1.0, (i+h, h))
        self.Wr = np.random.normal(0.0, 1.0, (i+h, h))
        self.Wh = np.random.normal(0.0, 1.0, (i+h, h))
        self.Wy = np.random.normal(0.0, 1.0, (h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def softmax(x):
        """
        softmax
        """
        exp = np.exp(x - np.max(x))
        return exp / exp.sum(axis=1, keepdims=True)

    @staticmethod
    def sigmoid(x):
        """
        sigmoid
        """
        return 1 / (1 + np.exp(-x))

    def forward(self, h_prev, x_t):
        """
        forward
        """
        ugate = self.sigmoid(
            np.matmul((np.concatenate(
                    (h_prev, x_t), axis=1)), self.Wz) + self.bz)
        resgate = self.sigmoid(
            np.matmul((np.concatenate(
                (h_prev, x_t), axis=1)), self.Wr) + self.br)
        cmc = np.tanh(
            np.matmul((np.concatenate(
                ((resgate * h_prev), x_t), axis=1)), self.Wh) + self.bh)
        h_next = h_prev * (1 - ugate) + ugate * cmc
        return h_next, self.softmax(self.by + np.matmul(h_next, self.Wy))
