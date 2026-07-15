from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from core import MLP, SoftmaxCrossEntropy, accuracy, mini_batches
from experiments.iris.iris_preprocess import prepare_iris

RESULTS_PATH = Path(__file__).resolve().parent / "iris_results.png"


def main() -> None:
    data = prepare_iris(test_frac=0.25, seed=0)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    class_names = data["class_names"]

    rng = np.random.default_rng(0)
    # linear output = logits; SoftmaxCrossEntropy applies softmax internally
    model = MLP([4, 16, 3], ["relu", "linear"], init="he", rng=rng)

    epochs = 200
    batch_size = 16
    lr = 0.1

    train_losses = []
    train_accs = []
    test_accs = []

    for _ in range(epochs):
        epoch_losses = []
        for xb, yb in mini_batches(X_train, y_train, batch_size=batch_size, shuffle=True, rng=rng):
            logits = model.forward(xb)
            epoch_losses.append(SoftmaxCrossEntropy.forward(logits, yb))
            model.backward(SoftmaxCrossEntropy.backward(logits, yb))
            model.update(lr)

        train_losses.append(float(np.mean(epoch_losses)))
        train_accs.append(accuracy(y_train, model.forward(X_train)))
        test_accs.append(accuracy(y_test, model.forward(X_test)))

    final_train = train_accs[-1]
    final_test = test_accs[-1]
    print(f"final train acc={final_train:.3f}  test acc={final_test:.3f}")

    # simple per-class accuracy on the test set
    test_logits = model.forward(X_test)
    test_pred = np.argmax(test_logits, axis=1)
    per_class = []
    for c, name in enumerate(class_names):
        mask = y_test == c
        per_class.append(float(np.mean(test_pred[mask] == y_test[mask])) if mask.any() else 0.0)
        print(f"  {name}: {per_class[-1]:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].plot(train_losses)
    axes[0].set_title("Train loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("softmax CE")

    axes[1].plot(train_accs, label="train")
    axes[1].plot(test_accs, label="test")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()

    axes[2].bar(class_names, per_class, color=["C0", "C1", "C2"])
    axes[2].set_title("Test accuracy by class")
    axes[2].set_ylim(0, 1.05)
    axes[2].tick_params(axis="x", rotation=15)

    fig.tight_layout()
    fig.savefig(RESULTS_PATH, dpi=150)
    plt.close(fig)
    print(f"saved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
