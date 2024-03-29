import warnings

warnings.filterwarnings("ignore")

from data import NounChunkDataLoader, NounChunkDataset
from perceptron import RecurrentPerceptron, RecurrentPerceptronStage
from config import THRESHOLD

from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
import argparse


def training(args):
    train_ds = NounChunkDataset.from_file(args.train_ds_path, _replace=False)
    test_ds = NounChunkDataset.from_file(args.test_ds_path, _replace=False)
    train_dl = NounChunkDataLoader(train_ds, batch_size=args.batch_size)
    test_dl = NounChunkDataLoader(test_ds, batch_size=args.batch_size)

    print("✨ Number of training examples: ", len(train_ds))
    print("✨ Number of testing examples: ", len(test_ds))

    model = RecurrentPerceptron(args.num_inputs)

    pbar = tqdm(range(args.epochs))
    for _ in pbar:
        train_batch_accuracies = list()
        train_batch_losses = list()
        test_batch_accuracies = list()
        test_batch_precisions = list()
        test_batch_recalls = list()
        test_batch_losses = list()

        for batch in train_dl:
            inputs = np.stack(batch["inputs"], axis=0)
            outputs = np.stack(batch["outputs"], axis=0)

            if inputs.shape[1] < 10:
                continue

            preds = model(inputs, RecurrentPerceptronStage.TRAIN)
            labels = (preds > THRESHOLD).astype(outputs.dtype)
            loss = model.loss_bwd(outputs, RecurrentPerceptronStage.TRAIN, 1.0)
            model.update(args.learning_rate, args.momentum_coeff)

            train_batch_losses.append(loss.item())
            train_batch_accuracies.append(
                accuracy_score(outputs.reshape((-1, 1)), labels.reshape((-1, 1)))
            )

        for batch in test_dl:
            inputs = np.stack(batch["inputs"], axis=0)
            outputs = np.stack(batch["outputs"], axis=0)

            if inputs.shape[1] < 10:
                continue

            preds = model(inputs, RecurrentPerceptronStage.EVAL)
            labels = (preds > THRESHOLD).astype(outputs.dtype)
            loss = model.loss_bwd(outputs, RecurrentPerceptronStage.EVAL)

            test_batch_losses.append(loss.item())
            test_batch_accuracies.append(
                accuracy_score(outputs.reshape((-1, 1)), labels.reshape((-1, 1)))
            )
            test_batch_precisions.append(
                precision_score(outputs.reshape((-1, 1)), labels.reshape((-1, 1)))
            )
            test_batch_recalls.append(
                recall_score(outputs.reshape((-1, 1)), labels.reshape((-1, 1)))
            )

        train_acc = 0
        test_acc = 0
        train_acc = np.stack(train_batch_accuracies, axis=0).mean().item()
        train_loss = np.stack(train_batch_losses, axis=0).mean().item()
        test_acc = np.stack(test_batch_accuracies, axis=0).mean().item()
        test_prec = np.stack(test_batch_precisions, axis=0).mean().item()
        test_recall = np.stack(test_batch_recalls, axis=0).mean().item()
        test_loss = np.stack(test_batch_losses, axis=0).mean().item()

        pbar.set_postfix(
            {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "test_prec": test_prec,
                "test_recall": test_recall,
            }
        )
    model.save_weights(args.model_ckpt_path)
    print(model.W, model.V, model.B)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Recurrent Perceptron Training")
    parser.add_argument("--train_ds_path", type=str, required=True)
    parser.add_argument("--test_ds_path", type=str, required=True)
    parser.add_argument("--model_ckpt_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_inputs", type=int, default=9)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--momentum_coeff", type=float, default=1e-1)

    args = parser.parse_args()
    training(args)
