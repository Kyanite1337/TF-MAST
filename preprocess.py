"""Load, preprocess, window, and split NinaPro DB5 data."""

import numpy as np
from scipy.io import loadmat
from pathlib import Path
from collections import Counter
from sklearn.preprocessing import StandardScaler

import config as cfg


def _load_one_mat(filepath: str) -> dict:
    """Load one .mat file and extract emg, restimulus, repetition."""
    mat = loadmat(filepath)
    return {
        "emg": mat["emg"].astype(np.float32),  # (T, 16)
        "restimulus": mat["restimulus"].ravel().astype(np.int64),  # (T,)
        "repetition": mat["repetition"].ravel().astype(np.int64),  # (T,)
    }


def load_all_raw(data_dir: str = None) -> list[dict]:
    """Load all .mat files from ninapro db5 directory.

    Returns list of dicts: [{"emg": (T,16), "restimulus": (T,), "repetition": (T,)}, ...]
    """
    if data_dir is None:
        data_dir = cfg.DATA_DIR
    data_dir = Path(data_dir)
    records = []
    for subj in range(1, cfg.N_SUBJECTS + 1):
        for ex in cfg.EXERCISES:
            fpath = data_dir / f"s{subj}" / f"S{subj}_{ex}_A1.mat"
            if fpath.exists():
                records.append(_load_one_mat(str(fpath)))
            else:
                print(f"[WARN] Missing file: {fpath}")
    print(f"[INFO] Loaded {len(records)} recordings from {cfg.N_SUBJECTS} subjects")
    return records


def sliding_window(records: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply sliding window to concatenated recordings.

    Returns:
        X: (N, C, W)  windows
        y: (N,)        labels per window (majority vote)
        r: (N,)        repetition per window (majority vote)
    """
    X_list, y_list, r_list = [], [], []

    win_len = cfg.WINDOW_LEN  # 40
    stride = cfg.WINDOW_STRIDE  # 20

    for rec in records:
        emg = rec["emg"]  # (T, 16)
        stim = rec["restimulus"]  # (T,)
        rep = rec["repetition"]  # (T,)

        T = emg.shape[0]
        for start in range(0, T - win_len + 1, stride):
            end = start + win_len
            window_emg = emg[start:end]  # (W, 16)
            window_stim = stim[start:end]
            window_rep = rep[start:end]

            # Majority-vote label (ignore 0 = rest transition)
            valid = window_stim > 0
            if valid.sum() > 0:
                lbl = Counter(window_stim[valid]).most_common(1)[0][0]
            else:
                continue  # skip pure-rest transition windows

            rep_vote = Counter(window_rep).most_common(1)[0][0]

            X_list.append(window_emg.T)  # → (16, W)
            y_list.append(lbl)
            r_list.append(rep_vote)

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    r = np.array(r_list, dtype=np.int64)
    print(f"[INFO] Sliding window: {X.shape}  (N, C, W)")
    return X, y, r


def normalize_fit_train(X: np.ndarray) -> StandardScaler:
    """Fit a per-channel Z-score scaler on X. Returns fitted scaler."""
    N, C, W = X.shape
    scalers = []
    for c in range(C):
        sc = StandardScaler()
        sc.fit(X[:, c, :].reshape(-1, 1))
        scalers.append(sc)
    return scalers


def normalize_apply(X: np.ndarray, scalers: list) -> np.ndarray:
    """Apply per-channel scalers to X."""
    X_out = X.copy()
    for c, sc in enumerate(scalers):
        X_out[:, c, :] = sc.transform(X[:, c, :].reshape(-1, 1)).reshape(X.shape[0], X.shape[2])
    return X_out


def split_by_repetition(X: np.ndarray, y: np.ndarray, r: np.ndarray):
    """Split into train (reps 1,3,4,6) and test (reps 2,5)."""
    train_mask = np.isin(r, cfg.TRAIN_REPS)
    test_mask = np.isin(r, cfg.TEST_REPS)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # Map labels to 0..52
    unique_labels = np.unique(np.concatenate([y_train, y_test]))
    label_map = {orig: idx for idx, orig in enumerate(unique_labels)}

    y_train = np.array([label_map[l] for l in y_train], dtype=np.int64)
    y_test = np.array([label_map[l] for l in y_test], dtype=np.int64)

    print(f"[INFO] Train: {X_train.shape[0]} windows | Test: {X_test.shape[0]} windows")
    return X_train, y_train, X_test, y_test


def load_and_preprocess(data_dir: str = None):
    """Full preprocessing pipeline. Returns X_train, y_train, X_test, y_test, scalers."""
    records = load_all_raw(data_dir)
    X, y, r = sliding_window(records)
    X_train, y_train, X_test, y_test = split_by_repetition(X, y, r)
    scalers = normalize_fit_train(X_train)
    X_train = normalize_apply(X_train, scalers)
    X_test = normalize_apply(X_test, scalers)
    return X_train, y_train, X_test, y_test, scalers


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, _ = load_and_preprocess()
    print(f"Train: {X_train.shape}, labels {y_train.shape}")
    print(f"Test:  {X_test.shape}, labels {y_test.shape}")
    print(f"Label distribution (train): {np.bincount(y_train)}")
