import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

N_CLASSES = 7
CLASS_LABELS = [1, 2, 3, 4, 5, 6, 7]
ID_COL = "id"
TARGET_COL = "target"
FOLDS_PATH = Path("folds.npy")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_train_test(train_path: str = "train.csv", test_path: str = "test.csv"):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def get_feature_columns(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    train_features = [c for c in train.columns if c not in (ID_COL, TARGET_COL)]
    test_features = [c for c in test.columns if c != ID_COL]
    assert train_features == test_features, "Train/test feature columns differ or are out of order."
    assert ID_COL not in train_features, "id leaked into model features."
    assert TARGET_COL not in test.columns, "test.csv unexpectedly contains target."
    return train_features


def encode_target(y: pd.Series | np.ndarray) -> np.ndarray:
    encoded = np.asarray(y, dtype=np.int64) - 1
    assert encoded.min() == 0 and encoded.max() == N_CLASSES - 1, "Target must encode to 0..6."
    assert sorted(np.unique(encoded).tolist()) == list(range(N_CLASSES)), "Missing encoded class."
    return encoded


def decode_target(y_encoded: np.ndarray) -> np.ndarray:
    decoded = np.asarray(y_encoded, dtype=np.int64) + 1
    assert set(np.unique(decoded)).issubset(set(CLASS_LABELS)), "Decoded labels outside 1..7."
    return decoded


def load_arrays(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]):
    X = train[feature_cols].to_numpy(dtype=np.float32)
    X_test = test[feature_cols].to_numpy(dtype=np.float32)
    y = encode_target(train[TARGET_COL].to_numpy())
    return X, y, X_test


def create_or_load_folds(y: np.ndarray, seed: int = 42, n_splits: int = 5) -> np.ndarray:
    if FOLDS_PATH.exists():
        folds = np.load(FOLDS_PATH)
        validate_folds(folds, y, n_splits=n_splits)
        return folds

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = np.full(len(y), -1, dtype=np.int8)
    dummy_X = np.zeros(len(y), dtype=np.int8)
    for fold_id, (_, val_idx) in enumerate(skf.split(dummy_X, y)):
        folds[val_idx] = fold_id
    validate_folds(folds, y, n_splits=n_splits)
    np.save(FOLDS_PATH, folds)
    return folds


def load_folds(y: np.ndarray, n_splits: int = 5) -> np.ndarray:
    assert FOLDS_PATH.exists(), "folds.npy does not exist. Create it once before model runs."
    folds = np.load(FOLDS_PATH)
    validate_folds(folds, y, n_splits=n_splits)
    return folds


def validate_folds(folds: np.ndarray, y: np.ndarray, n_splits: int = 5) -> None:
    assert folds.shape == (len(y),), "folds.npy must contain one fold id per train row."
    assert np.issubdtype(folds.dtype, np.integer), "fold ids must be integers."
    assert set(np.unique(folds).tolist()) == set(range(n_splits)), "fold ids must be exactly 0..4."
    for fold_id in range(n_splits):
        train_idx, val_idx = split_indices(folds, fold_id)
        assert len(set(train_idx).intersection(set(val_idx))) == 0, "Fold train/val overlap."
        assert len(train_idx) + len(val_idx) == len(y), "Fold split does not cover all rows."


def split_indices(folds: np.ndarray, fold_id: int):
    val_idx = np.where(folds == fold_id)[0]
    train_idx = np.where(folds != fold_id)[0]
    return train_idx, val_idx


def add_row_features(X: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        stats = np.column_stack([
            np.isnan(X).sum(axis=1),
            np.nanmean(X, axis=1),
            np.nanstd(X, axis=1),
            np.nanmin(X, axis=1),
            np.nanmax(X, axis=1),
            np.nansum(X, axis=1),
        ]).astype(np.float32)
    return np.hstack([X, stats]).astype(np.float32)


def class_weights(y: np.ndarray, exponent: float = 1.0) -> dict[int, float]:
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    weights = {int(c): float((total / (len(classes) * cnt)) ** exponent) for c, cnt in zip(classes, counts)}
    return weights


def sample_weights(y: np.ndarray, exponent: float = 1.0) -> np.ndarray:
    weights = class_weights(y, exponent=exponent)
    out = np.array([weights[int(v)] for v in y], dtype=np.float32)
    out /= out.mean()
    return out


def macro_f1(y_true: np.ndarray, proba: np.ndarray) -> float:
    return float(f1_score(y_true, np.argmax(proba, axis=1), average="macro"))


def per_class_f1(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    vals = f1_score(y_true, np.argmax(proba, axis=1), average=None, labels=np.arange(N_CLASSES))
    return {str(i + 1): float(v) for i, v in enumerate(vals)}


def confusion_matrix(y_true: np.ndarray, proba: np.ndarray) -> list[list[int]]:
    cm = sk_confusion_matrix(y_true, np.argmax(proba, axis=1), labels=np.arange(N_CLASSES))
    return cm.astype(int).tolist()


def assert_probability_array(proba: np.ndarray, expected_shape: tuple[int, int], name: str) -> None:
    assert proba.shape == expected_shape, f"{name} shape {proba.shape} != {expected_shape}."
    assert not np.isnan(proba).any(), f"{name} contains NaNs."
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4), f"{name} rows do not sum to 1."


def assert_oof_original_order(oof_seen: np.ndarray) -> None:
    assert oof_seen.dtype == bool, "oof_seen must be boolean."
    assert oof_seen.all(), "Some training rows were not assigned OOF probabilities."


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")
