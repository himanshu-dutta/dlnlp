import warnings

warnings.filterwarnings("ignore")

from data import NounChunkDataLoader, NounChunkDataset
from perceptron import RecurrentPerceptron, RecurrentPerceptronStage
from config import THRESHOLD

import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score


def crossval_splits(dataset: NounChunkDataset, splits: int):
    dataset_chunks = dataset.ksplits(splits)
    train_test_splits = list()
    for chunk_idx in range(len(dataset_chunks)):
        test_ds = dataset_chunks[chunk_idx]
        train_ds = NounChunkDataset(
            [], [], [], num_token_types=dataset_chunks[chunk_idx].num_token_types
        )
        for idx in range(len(dataset_chunks)):
            if idx == chunk_idx:
                continue
            train_ds.union(dataset_chunks[idx])
        train_test_splits.append((train_ds, test_ds))
    return train_test_splits


def plot_crossval_graph(stats: list[float], name: str, save_path: str = None):
    name = name.capitalize()
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(stats) + 1), stats, marker="o", linestyle="-")
    plt.title(f"{name} across Folds")
    plt.xlabel("Fold")
    plt.ylabel(name)
    plt.xticks(range(1, len(stats) + 1))
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    if save_path != None:
        plt.savefig(save_path, dpi=150)


def crossval_exp(args):
    ds = NounChunkDataset.from_file(args.ds_path, _replace=False)
    splits = crossval_splits(ds, args.folds)
    fold_metrics = {
        "accuracy": list(),
        "precision": list(),
        "recall": list(),
    }

    for fold_idx, split in enumerate(splits):
        train_ds, test_ds = split
        train_dl = NounChunkDataLoader(train_ds, batch_size=args.batch_size)
        test_dl = NounChunkDataLoader(test_ds, batch_size=args.batch_size)
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
                loss = model.loss_bwd(outputs, RecurrentPerceptronStage.TRAIN, 1.0)
                model.update(args.learning_rate, args.momentum_coeff)

                train_batch_losses.append(loss.item())
                labels = (preds > THRESHOLD).astype(outputs.dtype).reshape((-1, 1))
                train_batch_accuracies.append(
                    accuracy_score(
                        outputs.reshape((-1, 1)),
                        labels,
                    )
                )

            for batch in test_dl:
                inputs = np.stack(batch["inputs"], axis=0)
                outputs = np.stack(batch["outputs"], axis=0)

                if inputs.shape[1] < 10:
                    continue

                preds = model(inputs, RecurrentPerceptronStage.EVAL)
                labels = (preds > THRESHOLD).astype(outputs.dtype).reshape((-1, 1))
                loss = model.loss_bwd(outputs, RecurrentPerceptronStage.EVAL)

                test_batch_losses.append(loss.item())
                test_batch_accuracies.append(
                    accuracy_score(outputs.reshape((-1, 1)), labels)
                )
                test_batch_precisions.append(
                    precision_score(outputs.reshape((-1, 1)), labels)
                )
                test_batch_recalls.append(
                    recall_score(outputs.reshape((-1, 1)), labels)
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
                    "Fold": fold_idx + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "test_prec": test_prec,
                    "test_recall": test_recall,
                }
            )

        fold_metrics["accuracy"].append(test_acc)
        fold_metrics["precision"].append(test_prec)
        fold_metrics["recall"].append(test_recall)

    for metric_name, stats in fold_metrics.items():
        plot_crossval_graph(stats, metric_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Recurrent Perceptron Crossval Experiment")
    parser.add_argument("--ds_path", type=str, required=True)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_inputs", type=int, default=9)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--momentum_coeff", type=float, default=1e-1)

    args = parser.parse_args()
    crossval_exp(args)
