from typing import List
import numpy as np
from .common import Param


class SGD:
    def __init__(self, params: List[Param], lr: float, momentum_coeff: float = 0):
        self.params = params
        self.lr = lr
        self.momentum_coeff = momentum_coeff

    def step(self) -> None:
        for param in self.params:
            param.v = self.momentum_coeff * param.v + self.lr * param.grad
            param.d -= self.lr * param.v

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = np.zeros(shape=param.d.shape)
            param.v = np.zeros(shape=param.d.shape)
