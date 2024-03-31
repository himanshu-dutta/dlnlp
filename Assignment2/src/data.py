import re
import json
import string
import numpy as np
from operator import itemgetter

DTYPE = float
START_TOKEN_IDX = 0


class NounChunkDataset:
    def __init__(
        self,
        tokens: list[list[str]],
        pos_tags: list[list[int]],
        chunk_tags: list[list[int]] = None,
        num_token_types: int = 4,
        _filter: bool = False,
        _replace: bool = False,
    ):
        assert (not _filter) or (
            not _replace
        ), "can't have both filter and replace operation"

        self.tokens, self.pos_tags, self.chunk_tags = list(), list(), list()
        if _filter:
            for _t, _p, _c in zip(tokens, pos_tags, chunk_tags):
                if self._filter(_t, _p, _c):
                    self.tokens.append(_t)
                    self.pos_tags.append(_p)
                    self.chunk_tags.append(_c)
        if _replace:
            for _t, _p, _c in zip(tokens, pos_tags, chunk_tags):
                t, p, c = self._replace(_t, _p, _c)
                self.tokens.append(t)
                self.pos_tags.append(p)
                self.chunk_tags.append(c)
        else:
            self.tokens, self.pos_tags, self.chunk_tags = tokens, pos_tags, chunk_tags
        self.num_token_types = num_token_types

    @classmethod
    def from_file(
        cls,
        path: str,
        _filter: bool = False,
        _replace: bool = False,
    ):
        tokens, pos_tags, chunk_tags = cls._load_from_file(path)
        num_token_types = 0
        for pt in pos_tags:
            num_token_types = max([num_token_types, max(pt)])
        return cls(tokens, pos_tags, chunk_tags, num_token_types, _filter, _replace)

    @staticmethod
    def _replace(tokens, pos_tags, chunk_tags):
        def replace_pattern(pos_tags_s):
            pattern = r"2?3*1"
            matches = re.finditer(pattern, pos_tags_s)
            chunk_tags_re = ""
            prev_end = 0
            for match in matches:
                start, end = match.span()
                match_length = end - start
                chunk_tags_re += "1" * (start - prev_end)
                chunk_tags_re += "1" + "0" * (match_length - 1)
                prev_end = end
            chunk_tags_re += "1" * (len(pos_tags_s) - prev_end)
            return chunk_tags_re

        pos_mapping = {"NN": 1, "DT": 2, "JJ": 3, "OT": 4}
        pos_tags_s = "".join([str(v) for v in pos_tags])
        chunk_tags_re = replace_pattern(pos_tags_s)
        chunk_tags = [int(v) for v in list(chunk_tags_re)]

        return tokens, pos_tags, chunk_tags

    @staticmethod
    def _filter(tokens, pos_tags, chunk_tags):
        def replace_pattern(pos_tags_s):
            pattern = r"2?3*1"
            matches = re.finditer(pattern, pos_tags_s)
            chunk_tags_re = ""
            prev_end = 0
            for match in matches:
                start, end = match.span()
                match_length = end - start
                chunk_tags_re += "1" * (start - prev_end)
                chunk_tags_re += "1" + "0" * (match_length - 1)
                prev_end = end
            chunk_tags_re += "1" * (len(pos_tags_s) - prev_end)
            return chunk_tags_re

        pos_mapping = {"NN": 1, "DT": 2, "JJ": 3, "OT": 4}
        pos_tags_s = "".join([str(v) for v in pos_tags])
        chunk_tags_s = "".join([str(v) for v in chunk_tags])
        chunk_tags_re = replace_pattern(pos_tags_s)

        return chunk_tags_s == chunk_tags_re

    @staticmethod
    def _load_from_file(path: str):
        tokens, pos_tags, chunk_tags = list(), list(), list()
        with open(path, "r") as fp:
            for ln in fp.readlines():
                data = json.loads(ln)
                tokens.append(data["tokens"])
                pos_tags.append(data["pos_tags"])
                chunk_tags.append(data["chunk_tags"])
        return tokens, pos_tags, chunk_tags

    def ksplits(self, k: int, shuffle: bool = True):
        indices = np.arange(len(self), dtype=int)
        if shuffle:
            np.random.shuffle(indices)
        indices = indices.tolist()
        split_size = len(self) // k
        splits_indices = [
            indices[
                (idx * split_size) : (
                    len(indices) if idx == (k - 1) else ((idx + 1) * (split_size))
                )
            ]
            for idx in range(k)
        ]

        datasets = list()
        for split_indices in splits_indices:
            getter = itemgetter(*split_indices)
            tokens = getter(self.tokens)
            pos_tags = getter(self.pos_tags)
            chunk_tags = getter(self.chunk_tags) if self.chunk_tags != None else None
            dataset = NounChunkDataset(
                tokens, pos_tags, chunk_tags, self.num_token_types
            )
            datasets.append(dataset)

        return datasets

    def union(self, d: "NounChunkDataset"):
        assert d.num_token_types == self.num_token_types, "number of tokens must match"
        assert (self.chunk_tags != None) == (
            d.chunk_tags != None
        ), "either both `chunk_tag` attributes should be None or neither"

        self.tokens.extend(d.tokens)
        self.pos_tags.extend(d.pos_tags)
        if self.chunk_tags != None and d.chunk_tags != None:
            self.chunk_tags.extend(d.chunk_tags)

    def __len__(self):
        return len(self.tokens)

    def _onehot_encode(self, arr: np.ndarray, n: int = None, include_zero: bool = True):
        n = (n + 1) if n is not None else int(arr.max() + 1)
        oh = np.zeros((arr.size, n), dtype=DTYPE)
        oh[np.arange(arr.size), arr] = 1.0
        if not include_zero:
            oh = oh[:, 1:]
        return oh

    @staticmethod
    def mark_noun_chunks(noun_chunks: list[int]):
        if noun_chunks == None:
            return noun_chunks
        noun_chunk_str = "".join(map(str, noun_chunks))
        pattern_matches = re.finditer(r"10+", noun_chunk_str)
        output_list = [1] * len(noun_chunks)
        for match in pattern_matches:
            start = match.start()
            end = match.end()
            for i in range(start, end):
                output_list[i] = 0
        return output_list

    @staticmethod
    def mark_nouns(pos_tags: list[int], noun_chunks: list[int]):
        if noun_chunks == None:
            return noun_chunks
        for idx in range(len(pos_tags)):
            if pos_tags[idx] == 1:
                noun_chunks[idx] = 0
        return noun_chunks

    @staticmethod
    def preprocess(
        tokens: list[str], pos_tags: list[int], chunk_tags: list[int] = None
    ):
        alphanumerics = string.ascii_lowercase + string.ascii_uppercase + string.digits

        tokens_p, pos_tags_p, chunk_tags_p = (
            list(),
            list(),
            list() if chunk_tags else None,
        )

        for idx in range(len(tokens)):
            if any(ch in tokens[idx] for ch in alphanumerics):
                tokens_p.append(tokens[idx])
                pos_tags_p.append(pos_tags[idx])
                if chunk_tags is not None:
                    chunk_tags_p.append(chunk_tags[idx])
        return tokens_p, pos_tags_p, chunk_tags_p

    def __getitem__(self, idx: int):
        tokens, pos_tags, chunk_tags = (
            self.tokens[idx],
            self.pos_tags[idx],
            self.chunk_tags[idx] if self.chunk_tags is not None else None,
        )

        current_inp = self._onehot_encode(
            np.array(pos_tags, dtype=int),
            self.num_token_types,
            False,
        )
        previous_inp = self._onehot_encode(
            np.array([START_TOKEN_IDX] + pos_tags[:-1], dtype=int),
            self.num_token_types,
        )
        inputs = np.hstack((previous_inp, current_inp))
        outputs = (
            np.array(chunk_tags, dtype=DTYPE).reshape((-1, 1))
            if chunk_tags is not None
            else None
        )
        return {
            "tokens": tokens,
            "pos_tags": pos_tags,
            "chunk_tags": chunk_tags,
            "inputs": inputs,
            "outputs": outputs,
        }


class NounChunkDataLoader:
    def __init__(
        self,
        dataset: NounChunkDataset,
        batch_size: int,
        shuffle: bool = True,
        collate: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate = collate
        self.batched_indices = self._index()
        self.num_batches = len(self.batched_indices)

    def _index(self):
        indices = np.arange(len(self.dataset), dtype=int)
        if self.shuffle:
            np.random.shuffle(indices)
        batched_indices = list()
        for idx in range(0, len(self.dataset), self.batch_size):
            st = idx
            en = min(idx + self.batch_size, len(self.dataset))
            batched_indices.append(indices[st:en])
        return batched_indices

    def _collate(self, batch: list[dict]):
        assert len(batch) > 0, "batch should contain at least one example"

        batch_collated = dict()
        keys = batch[0].keys()
        for key in keys:
            val = [item[key] for item in batch]
            batch_collated[key] = val
        return batch_collated

    def _prepare_batch(self, idx: int):
        indices = self.batched_indices[idx]
        batch = list()
        for index in indices:
            itm = self.dataset[index]
            batch.append(itm)
        if self.collate:
            batch = self._collate(batch)
        return batch

    def __iter__(self):
        self.batch_index = 0
        return self

    def __next__(self):
        if self.batch_index >= self.num_batches:
            raise StopIteration
        else:
            batch = self._prepare_batch(self.batch_index)
            self.batch_index += 1
            return batch

    def __len__(self):
        return self.num_batches


if __name__ == "__main__":
    train_ds = NounChunkDataset.from_file("./data/train.jsonl")
    train_dl = NounChunkDataLoader(train_ds, 4, True)

    for batch in train_dl:
        print(
            batch["inputs"][0].shape,
            len(batch["inputs"]),
            len(batch["outputs"]),
        )
        break
