from __future__ import annotations

from pathlib import Path

import numpy as np

from core.utils import normalize, one_hot

DATA_PATH = Path(__file__).resolve().parent / "data" / "iris.csv"

CLASS_TO_IDX = {
    "setosa": 0,
    "versicolor": 1,
    "virginica": 2,
}


def load_iris(path: Path = DATA_PATH) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows = path.read_text().strip().splitlines()[1:]
    features = []
    labels = []
    for line in rows:
        parts = line.split(",")
        features.append([float(v) for v in parts[:4]])
        labels.append(CLASS_TO_IDX[parts[4].strip()])
    X = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=int)
    return X, y, ["setosa", "versicolor", "virginica"]


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_frac: float = 0.25,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    n_test = int(n * test_frac)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def prepare_iris(
    path: Path = DATA_PATH,
    test_frac: float = 0.25,
    seed: int = 0,
):
    # split first, then fit normalize stats on train only
    X, y, class_names = load_iris(path)
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_frac=test_frac, seed=seed)
    X_train, mean, std = normalize(X_train)
    X_test, _, _ = normalize(X_test, mean=mean, std=std)
    y_train_oh = one_hot(y_train, num_classes=len(class_names))
    y_test_oh = one_hot(y_test, num_classes=len(class_names))
    return {
        "X_train": X_train,
        "y_train": y_train,
        "y_train_oh": y_train_oh,
        "X_test": X_test,
        "y_test": y_test,
        "y_test_oh": y_test_oh,
        "class_names": class_names,
        "mean": mean,
        "std": std,
    }


if __name__ == "__main__":
    data = prepare_iris()
    print(
        f"train={data['X_train'].shape} test={data['X_test'].shape} "
        f"classes={data['class_names']}"
    )
    print("train mean ~0:", np.round(data["X_train"].mean(axis=0), 6))
    print("train std  ~1:", np.round(data["X_train"].std(axis=0), 6))
