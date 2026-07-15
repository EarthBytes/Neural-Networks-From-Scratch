from __future__ import annotations
import numpy as np
from core.activations import ReLU, get_activation
from core.layers import Dense

class MLP:
    # stack of Dense layers + activations with manual backprop and SGD

    def __init__(
        self,
        layer_sizes: list[int],
        activations: list[str],
        *,
        init: str = "xavier",
        rng: np.random.Generator | None = None,
    ) -> None:
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes needs at least [in_features, out_features]")
        if len(activations) != len(layer_sizes) - 1:
            raise ValueError("need one activation name per Dense layer")

        rng = rng or np.random.default_rng()
        self.layers = [
            Dense(layer_sizes[i], layer_sizes[i + 1], init=init, rng=rng)
            for i in range(len(layer_sizes) - 1)
        ]
        self.activations = [get_activation(name) for name in activations]
        self._cache: list[tuple[np.ndarray, np.ndarray, type]] = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cache = []
        a = x
        for layer, act in zip(self.layers, self.activations):
            z = layer.forward(a)
            out = act.forward(z)
            self._cache.append((z, out, act))
            a = out
        return a

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Walk activations then Dense layers in reverse.

        ReLU.backward expects the pre-activation z; the others cache the
        post-activation output from forward.
        """
        if not self._cache:
            raise RuntimeError("backward called before forward")

        for layer, (z, out, act) in reversed(list(zip(self.layers, self._cache))):
            cache = z if act is ReLU else out
            dout = act.backward(dout, cache)
            dout = layer.backward(dout)
        return dout

    def update(self, lr: float) -> None:
        for layer in self.layers:
            if layer.dW is None or layer.db is None:
                raise RuntimeError("update called before backward")
            layer.W -= lr * layer.dW
            layer.b -= lr * layer.db

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


def build_perceptron(
    in_features: int,
    *,
    rng: np.random.Generator | None = None,
) -> MLP:
    # single Dense + sigmoid — the linear model that can't solve XOR
    return MLP([in_features, 1], ["sigmoid"], init="xavier", rng=rng)
