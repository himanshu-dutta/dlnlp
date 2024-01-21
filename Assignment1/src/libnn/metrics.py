import numpy as np


def accuracy(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return (true == pred).mean()
