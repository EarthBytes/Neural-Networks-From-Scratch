from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from core import BinaryCrossEntropy, MLP, accuracy, build_perceptron
from experiments.xor.xor_data import make_xor_data, plot_xor_scatter

RESULTS_PATH = Path(__file__).resolve().parent / "xor_results.png"


def train_model(
    model: MLP,
    X: np.ndarray,
    y: np.ndarray,
    *,
    lr: float,
    epochs: int,
) -> list[float]:
    losses = []
    for _ in range(epochs):
        pred = model.forward(X)
        loss = BinaryCrossEntropy.forward(pred, y)
        losses.append(loss)
        model.backward(BinaryCrossEntropy.backward(pred, y))
        model.update(lr)
    return losses


def decision_boundary(model: MLP, ax, X: np.ndarray, y: np.ndarray, title: str) -> None:
    pad = 0.3
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict(grid).reshape(xx.shape)

    ax.contourf(xx, yy, probs, levels=50, cmap="RdBu", alpha=0.65)
    ax.contour(xx, yy, probs, levels=[0.5], colors="k", linewidths=1.2)
    plot_xor_scatter(X, y, ax=ax, title=title)


def main() -> None:
    rng = np.random.default_rng(0)
    X, y = make_xor_data(n_samples=400, noise=0.15, seed=0)

    perceptron = build_perceptron(2, rng=np.random.default_rng(1))
    mlp = MLP([2, 16, 1], ["relu", "sigmoid"], init="he", rng=rng)

    perc_losses = train_model(perceptron, X, y, lr=0.5, epochs=400)
    mlp_losses = train_model(mlp, X, y, lr=0.5, epochs=400)

    perc_acc = accuracy(y, (perceptron.predict(X) >= 0.5).astype(int))
    mlp_acc = accuracy(y, (mlp.predict(X) >= 0.5).astype(int))

    print(f"Perceptron  final loss={perc_losses[-1]:.4f}  acc={perc_acc:.3f}")
    print(f"MLP         final loss={mlp_losses[-1]:.4f}  acc={mlp_acc:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    decision_boundary(perceptron, axes[0], X, y, f"Perceptron (acc={perc_acc:.2f})")
    decision_boundary(mlp, axes[1], X, y, f"MLP (acc={mlp_acc:.2f})")

    axes[2].plot(perc_losses, label="perceptron")
    axes[2].plot(mlp_losses, label="MLP")
    axes[2].set_title("Training loss")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("BCE")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(RESULTS_PATH, dpi=150)
    plt.close(fig)
    print(f"saved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
