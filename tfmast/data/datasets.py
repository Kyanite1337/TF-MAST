from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class MAEDataset(Dataset):
    def __init__(self, x: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.x[idx]


class TFCDataset(MAEDataset):
    pass


class FineTuneDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


@dataclass
class LoaderBundle:
    mae: DataLoader
    tfc: DataLoader
    train: DataLoader
    test: DataLoader


def build_loaders(dataset, cfg) -> LoaderBundle:
    workers = int(cfg.train.num_workers)
    common = {"num_workers": workers, "pin_memory": workers > 0}
    if workers > 0:
        common["persistent_workers"] = True
    return LoaderBundle(
        mae=DataLoader(MAEDataset(dataset.x_train), batch_size=int(cfg.train.mae.batch_size), shuffle=True, drop_last=True, **common),
        tfc=DataLoader(TFCDataset(dataset.x_train), batch_size=int(cfg.train.tfc.batch_size), shuffle=True, drop_last=True, **common),
        train=DataLoader(FineTuneDataset(dataset.x_train, dataset.y_train), batch_size=int(cfg.train.finetune.batch_size), shuffle=True, **common),
        test=DataLoader(FineTuneDataset(dataset.x_test, dataset.y_test), batch_size=int(cfg.train.finetune.batch_size), shuffle=False, **common),
    )


def build_synthetic_loaders(num_train: int = 64, num_test: int = 32, batch_size: int = 8, num_classes: int = 53) -> LoaderBundle:
    rng = np.random.default_rng(42)
    x_train = rng.normal(size=(num_train, 16, 40)).astype(np.float32)
    y_train = np.arange(num_train, dtype=np.int64) % num_classes
    x_test = rng.normal(size=(num_test, 16, 40)).astype(np.float32)
    y_test = np.arange(num_test, dtype=np.int64) % num_classes
    return LoaderBundle(
        mae=DataLoader(MAEDataset(x_train), batch_size=batch_size, shuffle=True, drop_last=True),
        tfc=DataLoader(TFCDataset(x_train), batch_size=batch_size, shuffle=True, drop_last=True),
        train=DataLoader(FineTuneDataset(x_train, y_train), batch_size=batch_size, shuffle=True),
        test=DataLoader(FineTuneDataset(x_test, y_test), batch_size=batch_size, shuffle=False),
    )
