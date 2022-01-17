#!/usr/bin/env python3
"""
Sensitivity (True Positive Rate)
"""
import numpy as np


def sensitivity(confusion):
    """calculates the sensitivity for each class in a confusion matrix:
    - confusion is a confusion numpy.ndarray of shape (classes, classes)
where row indices represent the correct labels and column indices represent
the predicted labels.
    - classes is the number of classes
    * Returns: a numpy.ndarray of shape (classes,) containing the sensitivity
of each class
    """
    FP = confusion.sum(0) - np.diag(confusion)
    FN = confusion.sum(1) - np.diag(confusion)
    TP = np.diag(confusion)
    TN = confusion.sum() - (FP + FN + TP)
    return TP / (TP + FN)
