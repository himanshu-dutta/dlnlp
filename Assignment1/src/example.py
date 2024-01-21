from libnn.layers import Linear
from libnn.activations import Sigmoid, ReLU
from libnn.api import Sequential
from libnn.loss import BinaryCrossEntropy as BCELoss
from libnn.metrics import accuracy
from libnn.optim import SGD
import numpy as np


data = np.loadtxt("./data.csv", delimiter=",", dtype=np.float32)
Xs, Ys = data[:, :-1], data[:, -1]
num_inputs = Xs.shape[1]
num_epochs = 20

model = Sequential(
    Linear(num_inputs, 1024),
    Sigmoid(),
    Linear(1024, 1),
    Sigmoid(),
)
loss_fn = BCELoss()
optim = SGD(model.params, 1e-5)

for ep in range(num_epochs):
    outs = model(Xs)

    optim.zero_grad()
    loss = loss_fn(outs, Ys)
    loss_fn.backward()
    optim.step()

    acc = accuracy(Ys, outs > 0.5)

    print(f"Model Loss: {loss.item()}, Training accuracy: {acc.item()}")
