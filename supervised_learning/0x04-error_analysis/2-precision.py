#!/usr/bin/env python3
"""
Precision (positive predictive value)
"""
import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix:
    - `confusion` is a confusion numpy.ndarray of shape (classes, classes)
where row indices represent the correct labels and column indices
represent the predicted labels
    - `classes` is the number of classes
    * Returns: a numpy.ndarray of shape (classes,) containing the precision
of each class
    """
    TP = np.diag(confusion)  # Total positive
    FP = confusion.sum(0) - np.diag(confusion)  # False Positive
    FN = confusion.sum(1) - np.diag(confusion)
    TN = confusion.sum() - FP - FN - TP
    return TP / (TP + FP)
