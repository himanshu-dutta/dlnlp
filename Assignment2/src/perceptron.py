import numpy as np
import copy
import enum
import pickle

from data import NounChunkDataLoader, NounChunkDataset


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1.0 / np.sqrt(in_dim / 2.0)
    return np.random.normal(0, xavier_stddev, size)


class RecurrentPerceptronStage(enum.Enum):
    TRAIN: int = 1
    EVAL: int = 2


class RecurrentPerceptron:
    def __init__(self, num_inputs: int):
        self.num_inputs = num_inputs
        self._init_weights()
        self._init_state(reset_momentum=True)

    def _init_weights(self):
        self.W = xavier_init((self.num_inputs, 1))
        self.V = np.random.randn(1, 1) * 0.01
        self.B = xavier_init((1, 1))

    def _init_state(self, reset_momentum: bool = False):
        self.cache = {
            "inputs": None,
            "outputs": None,
            "dl_dw": None,
            "dl_db": None,
            "dl_dv": None,
            "loss": None,
            "num_steps": None,
            "mu_w": np.zeros_like(self.W),
            "mu_b": np.zeros_like(self.B),
            "mu_v": np.zeros_like(self.V),
        }

    def _reset_state(self, reset_momentum: bool = False):
        self.cache["inputs"] = None
        self.cache["outputs"] = None
        self.cache["dl_dw"] = None
        self.cache["dl_db"] = None
        self.cache["dl_dv"] = None
        self.cache["loss"] = None
        self.cache["num_steps"] = None

        if reset_momentum:
            self.cache["mu_w"] = np.zeros_like(self.W)
            self.cache["mu_b"] = np.zeros_like(self.B)
            self.cache["mu_v"] = np.zeros_like(self.V)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(
        self,
        inputs: np.ndarray,
        stage: RecurrentPerceptronStage = RecurrentPerceptronStage.EVAL,
    ):
        num_timesteps = inputs.shape[1]
        outputs = list()
        output = np.zeros((inputs.shape[0], 1), dtype=float)
        for t in range(num_timesteps):
            output = (inputs[:, t, :] @ self.W) + (self.V * output) - (self.B)
            output = self.sigmoid(output)
            outputs.append(copy.deepcopy(output))
        outputs = np.transpose(np.stack(outputs, axis=0), (1, 0, 2))
        self.cache["inputs"] = copy.deepcopy(inputs)
        self.cache["outputs"] = copy.deepcopy(outputs)
        return outputs

    def loss_bwd(
        self,
        trues: np.ndarray,
        stage: RecurrentPerceptronStage = RecurrentPerceptronStage.EVAL,
        clip_threshold: float = None,
    ):
        self.cache["loss"] = (1 / 2) * np.square((self.cache["outputs"] - trues)).mean(
            axis=1
        ).mean()

        if stage == RecurrentPerceptronStage.EVAL:
            return self.cache["loss"]

        num_timesteps = self.cache["inputs"].shape[1]

        dl_dw_cache = list()
        dl_db_cache = list()
        dl_dv_cache = list()

        dy_dw_cache = [np.zeros((self.cache["inputs"].shape[0], self.W.shape[0]))]
        dy_db_cache = [np.zeros((self.cache["inputs"].shape[0], self.B.shape[0]))]
        dy_dv_cache = [np.zeros((self.cache["inputs"].shape[0], self.V.shape[0]))]

        for ts in range(num_timesteps):
            dl_do = self.cache["outputs"][:, ts, :] - trues[:, ts, :]
            do_dy = self.cache["outputs"][:, ts, :] * (
                1.0 - self.cache["outputs"][:, ts, :]
            )
            dl_dy = dl_do * do_dy

            dy_dw = self.cache["inputs"][:, ts, :] + (self.V * dy_dw_cache[-1])
            dy_db = (np.ones((self.cache["inputs"].shape[0], self.B.shape[0])) * -1) + (
                self.V * dy_db_cache[-1]
            )
            dy_dv = (
                self.cache["outputs"][:, ts - 1, :]
                if ts > 0
                else np.zeros_like(self.cache["outputs"][:, ts, :])
            ) + (self.V * dy_db_cache[-1]).T

            dy_dw_cache.append(dy_dw)
            dy_db_cache.append(dy_db)
            dy_dv_cache.append(dy_dv)

            dl_dw = (dl_dy.T @ dy_dw).T
            dl_db = (dl_dy.T @ dy_db).T
            dl_dv = (dl_dy.T @ dy_dv).T

            if clip_threshold:
                dl_dw = np.clip(dl_dw, -clip_threshold, clip_threshold)
                dl_db = np.clip(dl_db, -clip_threshold, clip_threshold)
                dl_dv = np.clip(dl_dv, -clip_threshold, clip_threshold)

            dl_dw_cache.append(dl_dw)
            dl_db_cache.append(dl_db)
            dl_dv_cache.append(dl_dv)

        dl_dw = np.stack(dl_dw_cache, axis=0).mean(axis=0)
        dl_db = np.stack(dl_db_cache, axis=0).mean()
        dl_dv = np.stack(dl_dv_cache, axis=0).mean()

        self.cache["dl_dw"] = dl_dw
        self.cache["dl_db"] = dl_db
        self.cache["dl_dv"] = dl_dv

        return self.cache["loss"]

    def update(
        self,
        learning_rate: float,
        momentum_coeff: float = 0,
        reset_momentum: bool = False,
    ):
        self.cache["mu_w"] = (momentum_coeff * self.cache["mu_w"]) + (
            learning_rate * self.cache["dl_dw"]
        )
        self.cache["mu_b"] = (momentum_coeff * self.cache["mu_b"]) + (
            learning_rate * self.cache["dl_db"]
        )
        self.cache["mu_v"] = (momentum_coeff * self.cache["mu_v"]) + (
            learning_rate * self.cache["dl_dv"]
        )

        self.W = self.W - self.cache["mu_w"]
        self.B = self.B - self.cache["mu_b"]
        self.V = self.V - self.cache["mu_v"]
        self._reset_state(reset_momentum)

    def dump_weights(self):
        state = {
            "W": self.W.tolist(),
            "B": self.B.tolist(),
            "V": self.V.tolist(),
        }
        return state

    def save_weights(self, path: str):
        state = {
            "W": self.W.tolist(),
            "B": self.B.tolist(),
            "V": self.V.tolist(),
        }
        with open(path, "wb") as fp:
            pickle.dump(state, fp)

    def load_weights(self, path: str):
        with open(path, "rb") as fp:
            state = pickle.load(fp)
        self.W = np.array(state["W"]).reshape(self.W.shape)
        self.B = np.array(state["B"]).reshape(self.B.shape)
        self.V = np.array(state["V"]).reshape(self.V.shape)

    def __call__(
        self,
        inputs: np.ndarray,
        stage: RecurrentPerceptronStage = RecurrentPerceptronStage.EVAL,
    ):
        return self.forward(inputs, stage)


if __name__ == "__main__":
    model = RecurrentPerceptron(9)
    train_ds = NounChunkDataset.from_file("./data/train.jsonl")
    train_dl = NounChunkDataLoader(train_ds, 1, True)

    for batch in train_dl:
        inputs = np.stack(batch["inputs"], axis=0)
        outputs = np.stack(batch["outputs"], axis=0)
        preds = model(inputs, RecurrentPerceptronStage.TRAIN)
        loss = model.loss_bwd(outputs)
        model.update(1e-2)
        print(inputs.shape, outputs.shape, preds.shape, loss)
