import numpy as np


def metric_fp(y_true, y_pred):
    y_true = y_true.astype("uint8")
    # The try except is needed because when the metric is batched some batches
    # have one class only
    try:

        cm = np.array([[np.sum((y_pred < 0.5) & (y_true == 0)), np.sum((y_pred >= 0.5) & (y_true == 0))],
                       [np.sum((y_pred < 0.5) & (y_true == 1)), np.sum((y_pred >= 0.5) & (y_true == 1))]])

        # Calculate the false positive rate
        fp = cm[0][1] / (cm[0][1] + cm[1][1])

        return fp

    except ValueError:
        return np.nan
