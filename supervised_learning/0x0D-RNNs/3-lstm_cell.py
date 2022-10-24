#!/usr/bin/env python3
"""
LTSM
"""
import numpy as np


class LSTMCell:
    """
    LTSM
    """

    def __init__(self, i, h, o):
        """
        init
        """
        self.Wf = np.random.normal(0.0, 1.0, (i + h, h))
        self.Wu = np.random.normal(0.0, 1.0, (i + h, h))
        self.Wc = np.random.normal(0.0, 1.0, (i + h, h))
        self.Wo = np.random.normal(0.0, 1.0, (i + h, h))
        self.Wy = np.random.normal(0.0, 1.0, (h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
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

    def forward(self, h_prev, c_prev, x_t):
        """
        forward
        """
        forgate = self.sigmoid(
            np.matmul((np.concatenate(
                (h_prev, x_t), axis=1)), self.Wf) + self.bf)
        ugatee = self.sigmoid(
            np.matmul((np.concatenate(
                (h_prev, x_t), axis=1)), self.Wu) + self.bu)
        ics = np.tanh(
            np.matmul((np.concatenate(
                (h_prev, x_t), axis=1)), self.Wc) + self.bc)
        c_next = forgate * c_prev + ugatee * ics
        output_gate = self.sigmoid(
            np.matmul((np.concatenate(
                (h_prev, x_t), axis=1)), self.Wo) + self.bo)
        h_next = output_gate * np.tanh(c_next)
        y = self.softmax(self.by + np.matmul(h_next, self.Wy))
        return h_next, c_next, self.softmax(
            self.by + np.matmul(h_next, self.Wy))
