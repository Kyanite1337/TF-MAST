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
    mae_val: DataLoader
    tfc: DataLoader
    tfc_val: DataLoader
    train: DataLoader
    val: DataLoader
    test: DataLoader


def _split_train_val(x: np.ndarray, y: np.ndarray | None, fraction: float, seed: int):
    n = len(x)
    val_n = max(1, int(round(n * fraction))) if n > 1 else 1
    val_n = min(val_n, n - 1) if n > 1 else 1
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    val_idx, train_idx = idx[:val_n], idx[val_n:]
    if len(train_idx) == 0:
        train_idx = val_idx
    if y is None:
        return x[train_idx], None, x[val_idx], None
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def build_loaders(dataset, cfg) -> LoaderBundle:
    workers = int(cfg.train.num_workers)
    common = {"num_workers": workers, "pin_memory": workers > 0}
    if workers > 0:
        common["persistent_workers"] = True
    x_ssl_train, _, x_ssl_val, _ = _split_train_val(dataset.x_train, None, float(cfg.train.val_fraction), int(cfg.train.seed))
    x_ft_train, y_ft_train, x_ft_val, y_ft_val = _split_train_val(dataset.x_train, dataset.y_train, float(cfg.train.val_fraction), int(cfg.train.seed))
    return LoaderBundle(
        mae=DataLoader(MAEDataset(x_ssl_train), batch_size=int(cfg.train.mae.batch_size), shuffle=True, drop_last=True, **common),
        mae_val=DataLoader(MAEDataset(x_ssl_val), batch_size=int(cfg.train.mae.batch_size), shuffle=False, **common),
        tfc=DataLoader(TFCDataset(x_ssl_train), batch_size=int(cfg.train.tfc.batch_size), shuffle=True, drop_last=True, **common),
        tfc_val=DataLoader(TFCDataset(x_ssl_val), batch_size=int(cfg.train.tfc.batch_size), shuffle=False, **common),
        train=DataLoader(FineTuneDataset(x_ft_train, y_ft_train), batch_size=int(cfg.train.finetune.batch_size), shuffle=True, **common),
        val=DataLoader(FineTuneDataset(x_ft_val, y_ft_val), batch_size=int(cfg.train.finetune.batch_size), shuffle=False, **common),
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
        mae_val=DataLoader(MAEDataset(x_test), batch_size=batch_size, shuffle=False),
        tfc=DataLoader(TFCDataset(x_train), batch_size=batch_size, shuffle=True, drop_last=True),
        tfc_val=DataLoader(TFCDataset(x_test), batch_size=batch_size, shuffle=False),
        train=DataLoader(FineTuneDataset(x_train, y_train), batch_size=batch_size, shuffle=True),
        val=DataLoader(FineTuneDataset(x_test, y_test), batch_size=batch_size, shuffle=False),
        test=DataLoader(FineTuneDataset(x_test, y_test), batch_size=batch_size, shuffle=False),
    )
