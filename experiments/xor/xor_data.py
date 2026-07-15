from __future__ import annotations
import numpy as np

# four XOR corners with 0/1 labels (sigmoid + BCE friendly)
_XOR_CENTERS = np.array(
    [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ],
    dtype=float,
)
_XOR_LABELS = np.array([0.0, 1.0, 1.0, 0.0], dtype=float)


def make_xor_data(
    n_samples: int = 400,
    noise: float = 0.15,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    # scatter noisy points around each XOR corner
    rng = np.random.default_rng(seed)
    per_class = n_samples // 4
    extras = n_samples - per_class * 4

    xs = []
    ys = []
    for i, (center, label) in enumerate(zip(_XOR_CENTERS, _XOR_LABELS)):
        count = per_class + (1 if i < extras else 0)
        points = center + rng.normal(0.0, noise, size=(count, 2))
        xs.append(points)
        ys.append(np.full((count, 1), label))

    X = np.vstack(xs)
    y = np.vstack(ys)

    # shuffle so batches aren't ordered by cluster
    idx = rng.permutation(X.shape[0])
    return X[idx], y[idx]


def plot_xor_scatter(X: np.ndarray, y: np.ndarray, ax=None, title: str = "XOR data"):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    y_flat = y.reshape(-1)
    ax.scatter(X[y_flat < 0.5, 0], X[y_flat < 0.5, 1], c="C0", s=12, label="class 0", alpha=0.8)
    ax.scatter(X[y_flat >= 0.5, 0], X[y_flat >= 0.5, 1], c="C1", s=12, label="class 1", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="box")
    return ax


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    X, y = make_xor_data()
    print(f"X={X.shape}, y={y.shape}, mean label={y.mean():.3f}")
    plot_xor_scatter(X, y)
    plt.show()
