import numpy as np


def accuracy(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    true = true.reshape(pred.shape)
    assert true.shape == pred.shape, "shape should be same"
    return (true == pred).mean()


def precision(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    true = true.reshape(pred.shape)
    assert true.shape == pred.shape, "shape should be same"
    TP = ((pred == 1) & (true == 1)).sum()
    FP = ((pred == 1) & (true == 0)).sum()
    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    return precision


def recall(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    true = true.reshape(pred.shape)
    assert true.shape == pred.shape, "shape should be same"
    TP = ((pred == 1) & (true == 1)).sum()
    FN = ((pred == 0) & (true == 1)).sum()
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    return recall
