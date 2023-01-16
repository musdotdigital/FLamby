import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def metric(y_true, y_pred):
    y_true = y_true.type(torch.uint8)
    # The try except is needed because when the metric is batched some batches
    # have one class only
    try:
        return roc_auc_score(y_true, y_pred)
        # proposed modification in order to get a metric that calcs on center 2
        # (y=1 only on that center)
        return ((y_pred > 0.5) == y_true).mean()
    except ValueError:
        return np.nan
