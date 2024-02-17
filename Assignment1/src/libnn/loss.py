import numpy as np
from typing import Literal

from . import mem


class BinaryCrossEntropy:
    def __init__(
        self,
        reduction: Literal["mean"] | Literal["sum"] | Literal["none"] | None = "mean",
    ):
        self.reduction = reduction

    def __call__(self, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        self.prediction = prediction = np.clip(prediction, 1e-15, 1 - 1e-15)
        self.target = target
        loss = -(target * np.log(prediction) + (1 - target) * np.log(1 - prediction))

        if self.reduction == "none" or self.reduction == None:
            return loss
        if self.reduction == "sum":
            return np.sum(loss)
        return np.mean(loss)

    def backward(self, grads_in: np.ndarray = 1) -> None:
        grads_out = grads_in * (self.prediction - self.target)
        mem.backward_fn(grads_out)
        mem.backward_fn = mem.linear_backward
