#!/usr/bin/env python3
"""
The F-score or F-measure is a measure of a test's accuracy.
It is calculated from the precision and recall of the test.
"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score of a confusion matrix:
    - `confusionr` is a confusion numpy.ndarray of shape (classes, classes)
where row indices represent the correct labels and column indices
represent the predicted labels.
    - `classesr` is the number of classes
    * Returns: a numpy.ndarray of shape (classes,) containing the F1 score
of each class
    !!!Note: Using sensitivity = __import__('1-sensitivity').sensitivity and
precision = __import__('2-precision').precision create previously

    """
    return 2 / (pow(precision(confusion), -1) +
                pow(sensitivity(confusion), -1))
