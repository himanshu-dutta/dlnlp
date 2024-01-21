import numpy as np


class Param:
    def __init__(self, data: np.ndarray):
        self.d: np.ndarray = data
        self.grad: np.ndarray = None
