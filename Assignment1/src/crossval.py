from libnn.layers import Linear
from libnn.activations import Sigmoid, ReLU
from libnn.api import Sequential
from libnn.loss import BinaryCrossEntropy as BCELoss
from libnn.metrics import accuracy, precision, recall
from libnn.optim import SGD
from data.utils import train_test_split

import numpy as np
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

su = SMOTE(random_state=42)
data = np.loadtxt("./data.csv", delimiter=",", dtype=np.float32)
Xs, Ys = data[:, :-1], data[:, -1]
Xs, Ys = su.fit_resample(Xs, Ys)

n_splits = 4
skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
skf.get_n_splits(Xs, Ys)

for idx, (train_ids, test_ids) in enumerate(skf.split(Xs, Ys)):
    Xs_train, Ys_train = Xs[train_ids, :], Ys[train_ids]
    Xs_test, Ys_test = Xs[test_ids, :], Ys[test_ids]

    num_inputs = Xs_train.shape[1]
    num_epochs = 1000

    model = Sequential(
        Linear(num_inputs, 5),
        ReLU(),
        Linear(5, 1),
        Sigmoid(),
    )
    loss_fn = BCELoss()
    optim = SGD(model.params, lr=1e-4, momentum_coeff=1e-3)

    for ep in tqdm(range(num_epochs)):
        outs = model(Xs_train)

        optim.zero_grad()
        loss = loss_fn(outs, Ys_train)
        loss_fn.backward()
        optim.step()

        preds = model(Xs_test)

        optim.zero_grad()
        _ = loss_fn(preds, Ys_test)
        loss_fn.backward()
        optim.step()

        train_acc = accuracy(Ys_train, outs > 0.5)
        test_acc = accuracy(Ys_test, preds > 0.5)

        train_prec = precision(Ys_train, outs > 0.5)
        test_prec = precision(Ys_test, preds > 0.5)

        train_recall = recall(Ys_train, outs > 0.5)
        test_recall = recall(Ys_test, preds > 0.5)

    print(
        f"Fold[{idx+1}/{n_splits}] Model Loss: {loss.item():.4f}, Training accuracy: {train_acc.item():.4f}, Test accuracy: {test_acc.item():.4f}",
        f"Training precision: {train_prec.item():.4f}, Test precision: {test_prec.item():.4f}",
        f"Training recall: {train_recall.item():.4f}, Test recall: {test_recall.item():.4f}",
    )
