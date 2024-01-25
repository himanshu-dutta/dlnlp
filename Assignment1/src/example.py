from libnn.layers import Linear
from libnn.activations import Sigmoid, ReLU
from libnn.api import Sequential
from libnn.loss import BinaryCrossEntropy as BCELoss
from libnn.metrics import accuracy, precision, recall
from libnn.optim import SGD
from data.utils import train_test_split
import numpy as np


data = np.loadtxt("./data.csv", delimiter=",", dtype=np.float32)
training_data, test_data = train_test_split(data, 0.8)
Xs_train, Ys_train = training_data[:, :-1], training_data[:, -1]
Xs_test, Ys_test = test_data[:, :-1], test_data[:, -1]

num_inputs = Xs_train.shape[1]
num_epochs = 100

model = Sequential(
    Linear(num_inputs, 16),
    ReLU(),
    Linear(16, 1),
    Sigmoid(),
)
loss_fn = BCELoss()
optim = SGD(model.params, lr=1e-5, momentum_coeff=1e-3)

for ep in range(num_epochs):
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
        f"[{ep+1}/{num_epochs}] Model Loss: {loss.item():.4f}, Training accuracy: {train_acc.item():.4f}, Test accuracy: {test_acc.item():.4f}",
        f"Training precision: {train_prec.item():.4f}, Test precision: {test_prec.item():.4f}",
        f"Training recall: {train_recall.item():.4f}, Test recall: {test_recall.item():.4f}",
    )
