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
            grads_out = grads_in * (self._inp > 0).astype(float)
            self._grad_fn(grads_out)

        return _inner


class NormalAct:
    def __init__(self, inplace: bool = False):
        self.inplace = inplace

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = self.forward(x)
        self._grad_fn = mem.backward_fn
        mem.backward_fn = self._backward()
        return out

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._inp = x
        self._out = np.exp((-(x**2)) / 2)
        return self._out

    def _backward(self) -> Callable[[np.ndarray], np.ndarray]:
        def _inner(grads_in: np.ndarray) -> np.ndarray:
            grads_out = grads_in * (-self._out * self._inp)
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
            # grads_out = grads_in * (self._sigma * (1 - self._sigma))
            self._grad_fn(grads_in)

        return _inner
