from __future__ import annotations

import numpy as np


def set_seed(seed: int) -> np.random.Generator:
    # make training runs reproducible
    np.random.seed(seed)
    return np.random.default_rng(seed)


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    y = np.asarray(y).astype(int).reshape(-1)
    out = np.zeros((y.shape[0], num_classes), dtype=float)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # y_pred may be class indices or an (N, C) score / probability matrix
    yt = np.asarray(y_true).reshape(-1)
    yp = np.asarray(y_pred)
    # only argmax when there are multiple class scores (not a single sigmoid column)
    if yp.ndim > 1 and yp.shape[1] > 1:
        yp = np.argmax(yp, axis=1)
    else:
        yp = yp.reshape(-1)
    return float(np.mean(yt == yp))


def normalize(X: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None):
    # zero-mean, unit-variance; pass train-set stats when normalizing a holdout set
    X = np.asarray(X, dtype=float)
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
        # avoid divide-by-zero on constant features
        std = np.where(std < 1e-12, 1.0, std)
    return (X - mean) / std, mean, std


def mini_batches(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    rng: np.random.Generator | None = None,
):
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng = rng or np.random.default_rng()
        rng.shuffle(idx)

    # final batch may be smaller than batch_size
    for start in range(0, n, batch_size):
        batch = idx[start : start + batch_size]
        yield X[batch], y[batch]
