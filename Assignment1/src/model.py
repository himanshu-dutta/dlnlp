import numpy as np
import pickle

from libnn.layers import Linear
from libnn.activations import Sigmoid, ReLU
from libnn.api import Sequential
from libnn.loss import BinaryCrossEntropy as BCELoss
from libnn.optim import SGD

import numpy as np
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

import matplotlib.pyplot as plt


class PalindromeModel:
    def __init__(
        self,
        num_bits: int,
        threshold: float,
        learning_rate=0.1,
        momentum_coeff=0.9,
        nhidden=2,
    ):
        self.num_bits = num_bits
        self.threshold = threshold
        self.lr = learning_rate
        self.momentum_coeff = momentum_coeff
        self.nhidden = nhidden

        self.clf = Sequential(
            Linear(self.num_bits, self.nhidden, bias=False),
            ReLU(),
            Linear(self.nhidden, 1, bias=False),
            Sigmoid(),
        )

        self.loss_fn = BCELoss()
        self.optim = SGD(
            self.clf.params,
            lr=self.lr,
            momentum_coeff=momentum_coeff,
        )

    def fit(
        self,
        Xs: np.ndarray,
        Ys: np.ndarray,
        nepochs: int = 500,
        report_every_k_epochs: int = 50,
    ):
        progress_bar = tqdm(total=nepochs, desc="Training")

        for ep in range(nepochs):
            outs = self.clf(Xs)
            outs_cat = (outs >= self.threshold).astype(float)

            self.optim.zero_grad()
            loss = self.loss_fn(outs, Ys)
            self.loss_fn.backward()
            self.optim.step()

            train_acc = accuracy_score(Ys, outs_cat)
            train_prec = precision_score(Ys, outs_cat)
            train_recall = recall_score(Ys, outs_cat)

            if (ep + 1) % report_every_k_epochs == 0:
                message = f"Epoch [{ep+1}/{nepochs}], Model Loss: {loss.item():.4f}, Training accuracy: {train_acc:.4f}, Training precision: {train_prec:.4f}, Training recall: {train_recall:.4f}"
                progress_bar.set_description(message)
            progress_bar.update(1)
        progress_bar.close()

    def evaluate(self, Xs: np.ndarray, Ys: np.ndarray, plt_clf_report: bool = False):
        outs = self.clf(Xs)
        outs_cat = (outs >= self.threshold).astype(float)

        self.optim.zero_grad()
        eval_loss = self.loss_fn(outs, Ys)
        self.loss_fn.backward()

        eval_acc = accuracy_score(Ys, outs_cat)
        eval_prec = precision_score(Ys, outs_cat)
        eval_recall = recall_score(Ys, outs_cat)

        print(
            f"Evaluation Loss: {eval_loss.item():.4f}, Evaluation accuracy: {eval_acc:.4f}, Evaluation precision: {eval_prec:.4f}, Evaluation recall: {eval_recall:.4f}",
        )

        if plt_clf_report:
            print(classification_report(Ys, outs_cat))

        return {
            "accuracy": eval_acc,
            "precision": eval_prec,
            "recall": eval_recall,
        }

    def weights(self):
        print("Output-Hidden Weights:", self.clf.layers[2].params[0].d)
        print("Hidden-Input Weights:", self.clf.layers[0].params[0].d)

    def weight_visualization(self):
        hid_inp_w = self.clf.layers[0].params[0].d

        plt.figure(figsize=(10, 5))
        plt.title("Weight Vectors Heatmap")
        plt.imshow(hid_inp_w, cmap="hot", interpolation="nearest")
        plt.colorbar()
        plt.xlabel("Input Neurons")
        plt.ylabel("Hidden Neurons")
        plt.show()

    def predict(self, inps: np.ndarray):
        outs = self.clf(inps)
        outs_cat = (outs >= self.threshold).astype(float)

        self.optim.zero_grad()
        _ = self.loss_fn(outs, outs_cat)
        self.loss_fn.backward()

        return outs_cat

    def save(self, path: str):
        out_hid_w = self.clf.layers[2].params[0].d
        hid_inp_w = self.clf.layers[0].params[0].d

        weights = {
            "out_hid_w": out_hid_w,
            "hid_inp_w": hid_inp_w,
        }

        with open(path, "wb") as fp:
            pickle.dump(weights, fp)

    def load(self, path: str):
        with open(path, "rb") as fp:
            weights = pickle.load(fp)
            self.clf.layers[2].params[0].d = weights["out_hid_w"]
            self.clf.layers[0].params[0].d = weights["hid_inp_w"]
