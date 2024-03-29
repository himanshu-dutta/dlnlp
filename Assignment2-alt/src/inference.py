from data import NounChunkDataLoader, NounChunkDataset
from perceptron import RecurrentPerceptron, RecurrentPerceptronStage
from config import THRESHOLD

import numpy as np
import argparse


def prediction(tokens, pos_tags, args, model: RecurrentPerceptron = None):
    assert (model is not None) or (
        hasattr(args, "model_ckpt_path")
    ), "either model checkpoint path or the model instance must be provided for inference"

    if not model:
        model = RecurrentPerceptron(args.num_inputs)
        model.load_weights(args.model_ckpt_path)

    ds = NounChunkDataset(tokens, pos_tags, None)
    dl = NounChunkDataLoader(ds, batch_size=1)
    outputs = list()
    for batch in dl:
        inputs = np.stack(batch["inputs"], axis=0)
        preds = model(inputs, RecurrentPerceptronStage.EVAL)
        batch_outputs = (preds > THRESHOLD).astype(int).reshape((-1, 1))
        outputs.append(batch_outputs)
    outputs = np.stack(outputs, axis=0)
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Recurrent Perceptron Inference")
    parser.add_argument("--tokens", type=str, required=True)
    parser.add_argument("--pos_tags", type=str, required=True)
    parser.add_argument("--model_ckpt_path", type=str, required=True)
    parser.add_argument("--num_inputs", type=int, default=9)
    args = parser.parse_args()

    tokens = args.tokens.split(" ")
    pos_tags = [int(val) for val in args.pos_tags.split(" ")]
    outputs = prediction([tokens], [pos_tags], args)
    noun_chunk_tags = outputs[0].reshape((-1,)).tolist()

    print(tokens)
    print(pos_tags)
    print(noun_chunk_tags)
