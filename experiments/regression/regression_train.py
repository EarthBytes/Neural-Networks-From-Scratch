from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from core import MLP, MSE
from experiments.regression.regression_data import make_regression_data, train_val_split

RESULTS_PATH = Path(__file__).resolve().parent / "regression_results.png"


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    lr: float,
    epochs: int,
    seed: int,
) -> tuple[MLP, list[float], list[float]]:
    rng = np.random.default_rng(seed)
    model = MLP([1, 32, 32, 1], ["tanh", "tanh", "linear"], init="xavier", rng=rng)
    train_losses = []
    val_losses = []
    for _ in range(epochs):
        pred = model.forward(X_train)
        train_losses.append(MSE.forward(pred, y_train))
        model.backward(MSE.backward(pred, y_train))
        model.update(lr)
        val_losses.append(MSE.forward(model.forward(X_val), y_val))
    return model, train_losses, val_losses


def main() -> None:
    X, y = make_regression_data(n_samples=300, noise=0.1, seed=0)
    # scale x into roughly [-1, 1] so SGD stays well-behaved
    X_scaled = (X - np.pi) / np.pi
    X_train, y_train, X_val, y_val = train_val_split(X_scaled, y, val_frac=0.2, seed=0)

    learning_rates = [0.01, 0.05, 0.1]
    results = {}
    for lr in learning_rates:
        model, train_losses, val_losses = train_mlp(
            X_train, y_train, X_val, y_val, lr=lr, epochs=500, seed=1
        )
        results[lr] = (model, train_losses, val_losses)
        print(
            f"lr={lr:<5}  train MSE={train_losses[-1]:.4f}  val MSE={val_losses[-1]:.4f}"
        )

    best_lr = min(results, key=lambda lr: results[lr][2][-1])
    best_model = results[best_lr][0]
    print(f"best lr by val MSE: {best_lr}")

    # plot against the original x scale
    x_line = np.linspace(0.0, 2.0 * np.pi, 300).reshape(-1, 1)
    x_line_scaled = (x_line - np.pi) / np.pi
    y_hat = best_model.predict(x_line_scaled)
    y_true = np.sin(x_line)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # undo scaling for the scatter so the x-axis reads in radians
    X_train_plot = X_train * np.pi + np.pi
    X_val_plot = X_val * np.pi + np.pi
    axes[0].scatter(X_train_plot, y_train, s=12, alpha=0.6, label="train")
    axes[0].scatter(X_val_plot, y_val, s=12, alpha=0.6, label="val")
    axes[0].plot(x_line, y_true, "k--", linewidth=1.2, label="sin(x)")
    axes[0].plot(x_line, y_hat, "C3", linewidth=2, label=f"MLP (lr={best_lr})")
    axes[0].set_title("Fit vs noisy sine")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend()

    for lr, (_, train_losses, _) in results.items():
        axes[1].plot(train_losses, label=f"lr={lr}")
    axes[1].set_title("Training MSE by learning rate")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("MSE")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(RESULTS_PATH, dpi=150)
    plt.close(fig)
    print(f"saved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
