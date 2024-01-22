import numpy as np
from typing import Literal

from . import mem


class BinaryCrossEntropy:
    def __init__(
        self,
        reduction: Literal["mean"] | Literal["sum"] | Literal["none"] | None = "mean",
    ):
        self.reduction = reduction

    def __call__(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        input = input.reshape((-1))
        target = target.reshape((-1))
        assert (
            input.shape == target.shape
        ), f"input shape {input.shape} and target shape {target.shape} don't match"

        self.input = input
        self.target = target
        self.input = np.clip(self.input, 1e-7, 1 - 1e-7)
        term_0 = (1 - self.target) * np.log(1 - self.input + 1e-7)
        term_1 = self.target * np.log(self.input + 1e-7)
        loss = term_0 + term_1

        if self.reduction == "none" or self.reduction == None:
            return loss
        if self.reduction == "sum":
            return loss.sum()
        return loss.mean()

    def backward(self, grads_in: np.ndarray = 1) -> None:
        grads_out = grads_in * -(
            self.target * (1 / self.input) - (1 - self.target) * (1 / (1 - self.input))
        )
        # print(f"BCE Loss:: grads_out.shape: {grads_out.shape}")
        mem.backward_fn(grads_out)
        mem.backward_fn = mem.linear_backward
