from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from tfmast.config import load_config
from tfmast.data.preprocess import build_preprocessed_dataset, load_one_mat


def _write_db5_mat(root: Path, subject: int, exercise: int, samples: int = 240) -> Path:
    subject_dir = root / f"s{subject}"
    subject_dir.mkdir(parents=True, exist_ok=True)
    path = subject_dir / f"S{subject}_E{exercise}_A1.mat"
    emg = np.random.default_rng(subject * 10 + exercise).normal(size=(samples, 16)).astype(np.float32)
    restimulus = np.zeros(samples, dtype=np.int16)
    repetition = np.zeros(samples, dtype=np.int16)
    for rep in range(1, 7):
        start = (rep - 1) * (samples // 6)
        end = rep * (samples // 6)
        restimulus[start:end] = ((rep + exercise) % 52) + 1
        repetition[start:end] = rep
    savemat(path, {"emg": emg, "restimulus": restimulus, "rerepetition": repetition})
    return path


def test_load_one_mat_prefers_rerepetition(tmp_path):
    mat_path = _write_db5_mat(tmp_path / "data" / "ninapro_db5", subject=1, exercise=1)

    record = load_one_mat(mat_path, subject=1, exercise="E1")

    assert record.emg.shape == (240, 16)
    assert record.restimulus.shape == (240,)
    assert set(np.unique(record.repetition)) == {1, 2, 3, 4, 5, 6}
    assert record.subject == 1
    assert record.exercise == "E1"


def test_preprocess_splits_by_repetition_and_keeps_rest_class_when_configured(tmp_path):
    data_root = tmp_path / "data" / "ninapro_db5"
    for exercise in [1, 2, 3]:
        _write_db5_mat(data_root, subject=1, exercise=exercise)
    cfg = load_config(overrides={
        "data.root": str(data_root),
        "data.subjects": [1],
        "data.exercises": ["E1", "E2", "E3"],
        "data.class_mode": "53_with_rest",
        "data.window_ms": 200,
        "data.stride_ms": 100,
        "preprocess.notch.enabled": False,
        "preprocess.bandpass.enabled": False,
    })

    dataset = build_preprocessed_dataset(cfg, limit_subjects=1)

    assert dataset.x_train.shape[1:] == (16, 40)
    assert dataset.x_test.shape[1:] == (16, 40)
    assert set(np.unique(dataset.repetition_train)).issubset({1, 3, 4, 6})
    assert set(np.unique(dataset.repetition_test)).issubset({2, 5})
    assert dataset.num_classes == 53
    assert dataset.label_map[0] == 0


def test_preprocess_can_drop_rest_for_52_gesture_mode(tmp_path):
    data_root = tmp_path / "data" / "ninapro_db5"
    _write_db5_mat(data_root, subject=1, exercise=1)
    cfg = load_config(overrides={
        "data.root": str(data_root),
        "data.subjects": [1],
        "data.exercises": ["E1"],
        "data.class_mode": "52_gestures",
        "preprocess.notch.enabled": False,
        "preprocess.bandpass.enabled": False,
    })

    dataset = build_preprocessed_dataset(cfg, limit_subjects=1)

    assert dataset.num_classes == 52
    assert 0 not in dataset.label_map
    assert dataset.y_train.min() >= 0
    assert dataset.y_train.max() < 52


def test_missing_db5_file_error_names_expected_path(tmp_path):
    cfg = load_config(overrides={"data.root": str(tmp_path / "data" / "ninapro_db5"), "data.subjects": [1]})

    with pytest.raises(FileNotFoundError, match=r"S1_E1_A1\.mat"):
        build_preprocessed_dataset(cfg, limit_subjects=1)
