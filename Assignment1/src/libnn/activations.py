import numpy as np
from typing import Callable

from . import mem


class ReLU:
    def __init__(self, inplace: bool = False):
        self.inplace = inplace

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = self.forward(x)
        self._grad_fn = mem.backward_fn
        mem.backward_fn = self._backward()
        return out

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._inp = x
        return np.maximum(0, x)

    def _backward(self) -> Callable[[np.ndarray], np.ndarray]:
        def _inner(grads_in: np.ndarray) -> np.ndarray:
            grads_out = grads_in * np.where(self._inp > 0, 1, 0)
            # print(f"ReLU:: calling grad function: {self._grad_fn}")
            self._grad_fn(grads_out)

        return _inner


class Sigmoid:
    def __init__(self):
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = self.forward(x)
        self._grad_fn = mem.backward_fn
        mem.backward_fn = self._backward()
        return out

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._sigma = 1 / (1 + np.exp(-x))
        return self._sigma

    def _backward(self) -> Callable[[np.ndarray], np.ndarray]:
        def _inner(grads_in: np.ndarray) -> np.ndarray:
            grads_in = grads_in.reshape(self._sigma.shape)
            grads_out = grads_in * (self._sigma * (1 - self._sigma))
            # print(
            #     f"Sigmoid:: grads_in.shape: {grads_in.shape}, self._sigma.shape: {self._sigma.shape}, grads_out.shape: {grads_out.shape}"
            # )
            # print(f"Sigmoid:: calling grad function: {self._grad_fn}")
            self._grad_fn(grads_out)

        return _inner


class Softmax:
    def __init__(self, dim: int = None):
        self.dim: int = dim

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def forward(self):
        pass

    def _backward(self):
        pass
