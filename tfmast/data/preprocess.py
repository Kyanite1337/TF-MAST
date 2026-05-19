from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat
from scipy import signal

from tfmast.config import to_dict


@dataclass
class RawRecord:
    emg: np.ndarray
    restimulus: np.ndarray
    repetition: np.ndarray
    subject: int
    exercise: str
    path: Path


@dataclass
class PreprocessedDataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    repetition_train: np.ndarray
    repetition_test: np.ndarray
    subject_train: np.ndarray
    subject_test: np.ndarray
    exercise_train: np.ndarray
    exercise_test: np.ndarray
    label_map: dict[int, int]
    num_classes: int
    preprocessing_hash: str


def load_one_mat(path: str | Path, *, subject: int, exercise: str) -> RawRecord:
    path = Path(path)
    mat = loadmat(path)
    if "emg" not in mat or "restimulus" not in mat:
        raise KeyError(f"{path} must contain 'emg' and 'restimulus'")
    repetition_key = "rerepetition" if "rerepetition" in mat else "repetition"
    if repetition_key not in mat:
        raise KeyError(f"{path} must contain 'rerepetition' or 'repetition'")
    return RawRecord(
        emg=np.asarray(mat["emg"], dtype=np.float32),
        restimulus=np.asarray(mat["restimulus"]).reshape(-1).astype(np.int64),
        repetition=np.asarray(mat[repetition_key]).reshape(-1).astype(np.int64),
        subject=subject,
        exercise=exercise,
        path=path,
    )


def _expected_path(data_root: Path, subject: int, exercise: str) -> Path:
    return data_root / f"s{subject}" / f"S{subject}_{exercise}_A1.mat"


def _load_records(cfg: Any, limit_subjects: int | None = None) -> list[RawRecord]:
    root = Path(cfg.data.root)
    subjects = list(cfg.data.subjects)
    if limit_subjects is not None:
        subjects = subjects[:limit_subjects]
    records: list[RawRecord] = []
    for subject in subjects:
        for exercise in cfg.data.exercises:
            path = _expected_path(root, int(subject), str(exercise))
            if not path.exists():
                raise FileNotFoundError(f"Missing NinaPro DB5 file: {path}")
            records.append(load_one_mat(path, subject=int(subject), exercise=str(exercise)))
    return records


def _filter_emg(emg: np.ndarray, cfg: Any) -> np.ndarray:
    x = emg.astype(np.float32, copy=True)
    if cfg.preprocess.demean:
        x = x - x.mean(axis=0, keepdims=True)
    fs = float(cfg.data.sampling_rate)
    if cfg.preprocess.notch.enabled:
        try:
            b, a = signal.iirnotch(float(cfg.preprocess.notch.freq), float(cfg.preprocess.notch.quality), fs=fs)
            x = signal.filtfilt(b, a, x, axis=0).astype(np.float32)
        except Exception:
            # Old SciPy builds or very short synthetic signals can fail; keep deterministic preprocessing.
            pass
    if cfg.preprocess.bandpass.enabled:
        nyq = fs / 2.0
        low = float(cfg.preprocess.bandpass.low) / nyq
        high = min(float(cfg.preprocess.bandpass.high) / nyq, 0.999)
        if 0.0 < low < high:
            b, a = signal.butter(int(cfg.preprocess.bandpass.order), [low, high], btype="band")
            x = signal.filtfilt(b, a, x, axis=0).astype(np.float32)
    return x


def _majority(values: np.ndarray) -> int:
    return int(Counter(values.tolist()).most_common(1)[0][0])


def _window_records(records: list[RawRecord], cfg: Any) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    win = int(round(cfg.data.sampling_rate * cfg.data.window_ms / 1000))
    stride = int(round(cfg.data.sampling_rate * cfg.data.stride_ms / 1000))
    xs: list[np.ndarray] = []
    ys: list[int] = []
    reps: list[int] = []
    subjects: list[int] = []
    exercises: list[str] = []
    keep_rest = cfg.data.class_mode == "53_with_rest"

    for record in records:
        emg = _filter_emg(record.emg, cfg)
        for start in range(0, len(emg) - win + 1, stride):
            end = start + win
            labels = record.restimulus[start:end]
            if keep_rest:
                label = _majority(labels)
            else:
                non_rest = labels[labels > 0]
                if non_rest.size == 0:
                    continue
                label = _majority(non_rest)
            rep = _majority(record.repetition[start:end])
            xs.append(emg[start:end].T)
            ys.append(label)
            reps.append(rep)
            subjects.append(record.subject)
            exercises.append(record.exercise)

    if not xs:
        raise ValueError("No windows were produced; check class_mode, labels, and window settings.")
    meta = {
        "y": np.asarray(ys, dtype=np.int64),
        "repetition": np.asarray(reps, dtype=np.int64),
        "subject": np.asarray(subjects, dtype=np.int64),
        "exercise": np.asarray(exercises),
    }
    return np.stack(xs).astype(np.float32), meta


def _make_label_map(labels: np.ndarray, class_mode: str) -> dict[int, int]:
    if class_mode == "53_with_rest":
        return {i: i for i in range(53)}
    if class_mode == "52_gestures":
        return {i: i - 1 for i in range(1, 53)}
    unique = sorted(int(v) for v in np.unique(labels))
    return {label: idx for idx, label in enumerate(unique)}


def _zscore_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=(0, 2), keepdims=True)
    std = x_train.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((x_train - mean) / std).astype(np.float32), ((x_test - mean) / std).astype(np.float32)


def _hash_config(cfg: Any) -> str:
    relevant = {
        "data": to_dict(cfg.data),
        "preprocess": to_dict(cfg.preprocess),
    }
    blob = json.dumps(relevant, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def build_preprocessed_dataset(cfg: Any, *, limit_subjects: int | None = None) -> PreprocessedDataset:
    records = _load_records(cfg, limit_subjects=limit_subjects)
    x, meta = _window_records(records, cfg)
    train_mask = np.isin(meta["repetition"], np.asarray(cfg.data.train_reps))
    test_mask = np.isin(meta["repetition"], np.asarray(cfg.data.test_reps))
    if not train_mask.any() or not test_mask.any():
        raise ValueError("Repetition split produced an empty train or test set.")

    label_map = _make_label_map(meta["y"], cfg.data.class_mode)
    valid_labels = np.asarray([label in label_map for label in meta["y"]])
    train_mask &= valid_labels
    test_mask &= valid_labels

    x_train, x_test = x[train_mask], x[test_mask]
    if cfg.preprocess.zscore:
        x_train, x_test = _zscore_train_test(x_train, x_test)

    y_train = np.asarray([label_map[int(v)] for v in meta["y"][train_mask]], dtype=np.int64)
    y_test = np.asarray([label_map[int(v)] for v in meta["y"][test_mask]], dtype=np.int64)
    return PreprocessedDataset(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        repetition_train=meta["repetition"][train_mask],
        repetition_test=meta["repetition"][test_mask],
        subject_train=meta["subject"][train_mask],
        subject_test=meta["subject"][test_mask],
        exercise_train=meta["exercise"][train_mask],
        exercise_test=meta["exercise"][test_mask],
        label_map=label_map,
        num_classes=53 if cfg.data.class_mode == "53_with_rest" else 52,
        preprocessing_hash=_hash_config(cfg),
    )


def save_cache(dataset: PreprocessedDataset, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **dataset.__dict__)
