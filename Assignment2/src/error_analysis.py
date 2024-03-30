import warnings

warnings.filterwarnings("ignore")

from data import NounChunkDataLoader, NounChunkDataset
from perceptron import RecurrentPerceptron, RecurrentPerceptronStage
from config import THRESHOLD

from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
import argparse
import json


def error_analysis(args):
    test_ds = NounChunkDataset.from_file(args.test_ds_path, _replace=True)
    test_dl = NounChunkDataLoader(test_ds, batch_size=args.batch_size)
    error_datapoints = list()
    model = RecurrentPerceptron(args.num_inputs)
    model.load_weights(args.model_ckpt_path)
    print("✨ Number of testing examples: ", len(test_ds))
    for batch in tqdm(test_dl):
        inputs = np.stack(batch["inputs"], axis=0)
        outputs = np.stack(batch["outputs"], axis=0)
        preds = model(inputs, RecurrentPerceptronStage.EVAL)
        labels = (preds > THRESHOLD).astype(int).reshape((-1, 1))
        acc = accuracy_score(outputs.reshape((-1, 1)), labels.reshape((-1, 1)))
        if acc < args.error_threshold:
            batch.pop("inputs")
            batch.pop("outputs")
            batch["prediction"] = labels.reshape((-1,)).tolist()
            error_datapoints.append(batch)

    with open(args.save_path, "w") as fp:
        json.dump(error_datapoints, fp, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Recurrent Perceptron Training")
    parser.add_argument("--test_ds_path", type=str, required=True)
    parser.add_argument("--model_ckpt_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--error_threshold", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_inputs", type=int, default=9)

    args = parser.parse_args()
    error_analysis(args)
