# =============================================================================
# data.py — ImageNet-64 loader (downsampled ImageNet, 1,281,167 train images).
#
# Expects the official train_data_batch_1..10 / val_data pickle files.
# Images are stored flat uint8 (3*64*64) and labels are 1-indexed.
# =============================================================================

import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


class ImageNet64(Dataset):
    def __init__(self, root, train=True):
        files = ([f"train_data_batch_{i}" for i in range(1, 11)] if train
                 else ["val_data"])
        xs, ys = [], []
        for f in files:
            p = os.path.join(root, f)
            if not os.path.exists(p):
                raise FileNotFoundError(f"missing {p}")
            with open(p, "rb") as fh:
                d = pickle.load(fh, encoding="bytes")
            xs.append(d[b"data" if b"data" in d else "data"])
            ys.append(d[b"labels" if b"labels" in d else "labels"])

        self.x = np.concatenate(xs).astype(np.uint8)
        self.y = np.concatenate(ys).astype(np.int64) - 1        # -> 0..999
        assert self.y.min() >= 0 and self.y.max() < 1000

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        img = self.x[i].reshape(3, 64, 64).astype(np.float32)
        img = img / 127.5 - 1.0                                 # -> [-1, 1]
        return torch.from_numpy(img), int(self.y[i])
