from typing import List, Callable
import numpy as np

from . import mem
from .common import Param


class Linear:
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        """
        w: (of, if)
        b: (of, )
        inp: (BS, if)
        out: (BS, of)
        """
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self._init_params()

    def _init_params(self) -> None:
        self.w = Param(np.random.normal(size=(self.out_features, self.in_features)))
        if self.bias:
            self.b = Param(np.random.normal(size=(self.out_features)))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        out = self.forward(x)
        self._grad_fn = mem.backward_fn
        mem.backward_fn = self._backward()
        return out

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.inp = x
        out = self.inp @ self.w.d.T
        if self.bias:
            out += self.b.d
        return out

    def _backward(self) -> Callable[[np.ndarray], np.ndarray]:
        def _inner(grads_in: np.ndarray) -> np.ndarray:
            # print(
            #     f"Linear:: grads_in.shape: {grads_in.shape}, self.inp.shape: {self.inp.shape}, self.w.d.shape: {self.w.d.shape}",
            # )

            w_grad = grads_in.T @ self.inp
            b_grad = grads_in
            grads_out = grads_in @ self.w.d

            self.w.grad = w_grad.mean(axis=0)
            if self.bias:
                self.b.grad = b_grad.mean(axis=0)

            # print(f"Linear:: calling grad function: {self._grad_fn}")
            self._grad_fn(grads_out)

        return _inner

    @property
    def params(self) -> List[np.ndarray]:
        if self.bias:
            return [self.w, self.b]
        return [self.w]


class RNNUnit:
    def __init__(self, in_features, out_features):
        pass

    def forward(self):
        pass

    def _backward(self):
        pass


class RNN:
    def __init__(self, in_features, out_features):
        pass

    def forward(self):
        pass

    def _backward(self):
        pass
