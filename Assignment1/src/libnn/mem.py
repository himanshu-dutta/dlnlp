from typing import Callable
import numpy as np


def linear_backward(grad_in: np.ndarray) -> np.ndarray:
    return grad_in


global backward_fn
backward_fn: Callable[[np.ndarray], np.ndarray] = linear_backward
