import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def metric_fed(y_true, y_pred):
    y_true = y_true.type(torch.uint8)
    # The try except is needed because when the metric is batched some batches
    # have one class only
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        return np.nan
