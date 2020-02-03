import pdb

import numpy as np
from sklearn.metrics import (classification_report, f1_score, precision_score,
                             recall_score)


def score(gold, pred, average=None):
    """
    Returns a list if average=None, a float number otherwise
    average:
        - micro
        - macro
        - average
    """

    precision = precision_score(gold, pred, average=average)
    recall = recall_score(gold, pred, average=average)
    f1 = f1_score(gold, pred, average=average)

    return precision, recall, f1
