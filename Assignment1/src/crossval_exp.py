from libnn.layers import Linear
from libnn.activations import Sigmoid, ReLU
from libnn.api import Sequential
from libnn.loss import BinaryCrossEntropy as BCELoss
from libnn.optim import SGD

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
)
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
from sklearn.utils import shuffle

seed = 2
np.random.seed(seed)

su = SMOTE(random_state=seed)
data = np.loadtxt("./data.csv", delimiter=",", dtype=int)
data = data.astype(float)
Xs, Ys = data[:, :-1], data[:, -1]
Xs, Ys = su.fit_resample(Xs, Ys)
Xs, Ys = shuffle(Xs, Ys)

THRESHOLD = 0.4

n_splits = 4
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
skf.get_n_splits(Xs, Ys)

for idx, (train_ids, test_ids) in enumerate(skf.split(Xs, Ys)):
    Xs_train, Ys_train = Xs[train_ids, :], Ys[train_ids]
    Xs_test, Ys_test = Xs[test_ids, :], Ys[test_ids]

    Ys_train = Ys_train.reshape((-1, 1))
    Ys_test = Ys_test.reshape((-1, 1))

    num_inputs = Xs_train.shape[1]
    num_epochs = 1000

    model = Sequential(
        Linear(num_inputs, 2, bias=False),
        ReLU(),
        Linear(2, 1, bias=False),
        Sigmoid(),
    )
    loss_fn = BCELoss()
    optim = SGD(model.params, lr=0.1, momentum_coeff=0.9)

    for ep in tqdm(range(num_epochs)):
        outs = model(Xs_train)
        outs_cat = outs > THRESHOLD

        optim.zero_grad()
        loss = loss_fn(outs, Ys_train)
        loss_fn.backward()
        optim.step()

    preds = model(Xs_test)
    preds_cat = preds > THRESHOLD

    optim.zero_grad()
    _ = loss_fn(preds, Ys_test)
    loss_fn.backward()

    train_acc = accuracy_score(Ys_train, outs_cat)
    test_acc = accuracy_score(Ys_test, preds_cat)

    train_prec = precision_score(Ys_train, outs_cat)
    test_prec = precision_score(Ys_test, preds_cat)

    train_recall = recall_score(Ys_train, outs_cat)
    test_recall = recall_score(Ys_test, preds_cat)

    print(
        f"Fold[{idx+1}/{n_splits}] Model Loss: {loss.item():.4f}, Training accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}",
        f"Training precision: {train_prec:.4f}, Test precision: {test_prec:.4f}",
        f"Training recall: {train_recall:.4f}, Test recall: {test_recall:.4f}",
    )
    print(classification_report(Ys_test, preds_cat))
    print(model.layers[0].params[0].d, model.layers[2].params[0].d)
    input()
