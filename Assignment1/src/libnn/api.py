import numpy as np
from typing import List


class Sequential:
    def __init__(self, *args):
        self.layers = list(args)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y = x
        for lay in self.layers:
            y = lay(y)
        return y

    @property
    def params(self) -> List[np.ndarray]:
        p_ = []
        for layer in self.layers:
            if hasattr(layer, "params"):
                p_ += layer.params
        return p_
