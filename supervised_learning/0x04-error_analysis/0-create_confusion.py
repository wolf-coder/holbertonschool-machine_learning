#!/usr/bin/env python3
"""
confusion marix
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
Function that creates a confusion matrix:
    labels is a one-hot numpy.ndarray of shape (m, classes) containing
the correct labels for each data point
        m is the number of data points
        classes is the number of classes
    logits is a one-hot numpy.ndarray of shape (m, classes)
containing the predicted labels

    Returns: a confusion numpy.ndarray of shape (classes, classes)
with row indices representing the correct labels and column indices
representing the predicted labels
    """
    return np.matmul(labels.T, logits)
