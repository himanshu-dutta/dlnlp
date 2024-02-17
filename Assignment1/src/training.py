from model import PalindromeModel
import argparse

from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def load_data(path: str, seed: int):
    su = SMOTE(random_state=seed)
    data = np.loadtxt(path, delimiter=",", dtype=int)
    data = data.astype(float)
    Xs, Ys = data[:, :-1], data[:, -1]
    Xs, Ys = su.fit_resample(Xs, Ys)
    Xs, Ys = shuffle(Xs, Ys)
    Ys = Ys.reshape((-1, 1))
    return Xs, Ys


def main(args):
    model = PalindromeModel(
        10,
        args.threshold,
        learning_rate=args.lr,
        momentum_coeff=args.momentum,
        nhidden=2,
    )

    Xs, Ys = load_data(args.data_csv, args.seed)
    Xs_train, Xs_test, Ys_train, Ys_test = train_test_split(
        Xs, Ys, test_size=args.test_split, stratify=Ys
    )

    model.fit(
        Xs_train,
        Ys_train,
        nepochs=args.num_epochs,
    )

    model.evaluate(
        Xs_test,
        Ys_test,
        plt_clf_report=True,
    )

    model.weights()
    model.weight_visualization()

    model.save(args.save_path + "/model.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_csv", type=str, required=True)
    parser.add_argument("-e", "--num_epochs", type=int, default=500)
    parser.add_argument("-l", "--lr", type=float, default=0.1)
    parser.add_argument("-m", "--momentum", type=float, default=0.9)
    parser.add_argument("-t", "--threshold", type=float, default=0.4)
    parser.add_argument("-r", "--seed", type=int, default=2)
    parser.add_argument("-v", "--test_split", type=float, default=0.25)

    parser.add_argument("-s", "--save_path", type=str)

    args = parser.parse_args()

    np.random.seed(args.seed)

    main(args)
