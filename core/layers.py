from __future__ import annotations
import numpy as np

class Dense:
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        init: str = "xavier",
        rng: np.random.Generator | None = None,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features
        rng = rng or np.random.default_rng()

        if init == "xavier":
            std = np.sqrt(2.0 / (in_features + out_features))
        elif init == "he":
            std = np.sqrt(2.0 / in_features)
        else:
            raise ValueError(f"Unknown init {init!r}; use 'xavier' or 'he'")

        self.W = rng.normal(0.0, std, size=(in_features, out_features))
        self.b = np.zeros(out_features, dtype=float)

        self.dW: np.ndarray | None = None
        self.db: np.ndarray | None = None
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.W + self.b

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self._x is None:
            raise RuntimeError("backward called before forward")

        # dW = x.T @ dout ; db = sum over batch ; dx = dout @ W.T
        self.dW = self._x.T @ dout
        self.db = np.sum(dout, axis=0)
        return dout @ self.W.T

    def params_and_grads(self):
        yield self.W, self.dW
        yield self.b, self.db
