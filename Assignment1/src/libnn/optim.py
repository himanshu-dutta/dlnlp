from typing import List
import numpy as np
from .common import Param


class SGD:
    def __init__(self, params: List[Param], lr: float):
        self.params = params
        self.lr = lr

    def step(self) -> None:
        for param in self.params:
            param.d -= self.lr * param.grad

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = np.zeros(shape=param.d.shape)
