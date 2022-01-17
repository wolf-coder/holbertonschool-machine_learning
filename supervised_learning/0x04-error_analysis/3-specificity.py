#!/usr/bin/env python3
"""
specificity (True Negative Rate)
Note:
When there are more than two classes in a confusion matrix, specificity
is not a useful metric as there are inherently more actual negatives than
actual positives. It is much better to use sensitivity (recall) and precision.
"""
import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix:
    - `confusion` is a confusion numpy.ndarray of shape (classes, classes)
where row indices represent the correct labels and column indices
represent the predicted labels
    - `classes` is the number of classes
    * Returns: a numpy.ndarray of shape (classes,)containing the specificity
of each class
    """
    TP = np.diag(confusion)
    FP = confusion.sum(0) - np.diag(confusion)
    FN = confusion.sum(1) - np.diag(confusion)
    TN = confusion.sum() - (FP + FN + TP)
    return TN / (TN + FP)
