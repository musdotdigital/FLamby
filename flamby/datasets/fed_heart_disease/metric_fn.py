import numpy as np


def metric_fn(y_true, y_pred):
    y_true = y_true.astype("uint8")
    # The try except is needed because when the metric is batched some batches
    # have one class only
    try:

        cm = np.array([[np.sum((y_pred < 0.5) & (y_true == 0)), np.sum((y_pred >= 0.5) & (y_true == 0))],
                       [np.sum((y_pred < 0.5) & (y_true == 1)), np.sum((y_pred >= 0.5) & (y_true == 1))]])

        # Calculate the false negative rate
        fn = cm[1][0] / (cm[1][0] + cm[1][1])

        return fn

    except ValueError:
        return np.nan
