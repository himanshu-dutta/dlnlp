import numpy as np


def train_test_split(Xs: np.ndarray, train_pct: float = 0.6):
    num_train = int(Xs.shape[0] * train_pct)
    np.random.shuffle(Xs)
    training, test = Xs[:num_train, :], Xs[num_train:, :]
    return training, test
