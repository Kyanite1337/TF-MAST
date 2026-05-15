"""PyTorch Datasets for three-stage training."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MAEDataset(Dataset):
    """Stage 1: returns unlabeled windows for masked autoencoding."""

    def __init__(self, X: np.ndarray):
        self.X = torch.from_numpy(X)  # (N, C, W)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


class TFCDataset(Dataset):
    """Stage 2: returns each window twice for Siamese contrastive streams."""

    def __init__(self, X: np.ndarray):
        self.X = torch.from_numpy(X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        return x, x  # (x_time, x_freq) - augmentations applied in training loop


class FineTuneDataset(Dataset):
    """Stage 3: returns windows with labels for supervised learning."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_loaders(X_train, y_train, X_test, y_test):
    """Create all DataLoaders needed for the three stages."""

    # Stage 1 & 2 use only unlabeled training data
    from config import MAE_BATCH_SIZE, TFC_BATCH_SIZE, FT_BATCH_SIZE, NUM_WORKERS

    mae_loader = DataLoader(
        MAEDataset(X_train),
        batch_size=MAE_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    tfc_loader = DataLoader(
        TFCDataset(X_train),
        batch_size=TFC_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    train_loader = DataLoader(
        FineTuneDataset(X_train, y_train),
        batch_size=FT_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        FineTuneDataset(X_test, y_test),
        batch_size=FT_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return mae_loader, tfc_loader, train_loader, test_loader
