from __future__ import annotations

import numpy as np


def make_regression_data(
    n_samples: int = 300,
    noise: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    # target: y = sin(x) + noise, x in [0, 2π]
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 2.0 * np.pi, size=(n_samples, 1))
    y = np.sin(x) + noise * rng.normal(size=(n_samples, 1))
    return x, y


def train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    val_frac: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    n_val = int(n * val_frac)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def plot_regression_scatter(X: np.ndarray, y: np.ndarray, ax=None, title: str = "regression data"):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(X.ravel(), y.ravel(), s=12, alpha=0.7, label="data")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    X, y = make_regression_data()
    print(f"X={X.shape}, y={y.shape}")
    plot_regression_scatter(X, y)
    plt.show()
