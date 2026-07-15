from __future__ import annotations
import numpy as np

class Sigmoid:
    # XOR output activation (pairs with binary cross-entropy)

    @staticmethod
    def forward(z: np.ndarray) -> np.ndarray:
        # Stable form: handle positive and negative z separately so exp doesn't overflow
        out = np.empty_like(z, dtype=float)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        exp_z = np.exp(z[~pos])
        out[~pos] = exp_z / (1.0 + exp_z)
        return out

    @staticmethod
    def backward(dout: np.ndarray, cache: np.ndarray) -> np.ndarray:
        # cache = sigmoid(z) from forward
        return dout * cache * (1.0 - cache)


class ReLU:
    # common hidden-layer activation
    @staticmethod
    def forward(z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    @staticmethod
    def backward(dout: np.ndarray, cache: np.ndarray) -> np.ndarray:
        # cache = pre-activation z from forward
        dz = dout.copy()
        dz[cache <= 0] = 0.0
        return dz


class Tanh:
    # smooth hidden activation - useful for regression
    @staticmethod
    def forward(z: np.ndarray) -> np.ndarray:
        return np.tanh(z)

    @staticmethod
    def backward(dout: np.ndarray, cache: np.ndarray) -> np.ndarray:
        # cache = tanh(z) from forward
        return dout * (1.0 - cache**2)


class Linear:
    # identity — used for regression outputs

    @staticmethod
    def forward(z: np.ndarray) -> np.ndarray:
        return z

    @staticmethod
    def backward(dout: np.ndarray, cache: np.ndarray) -> np.ndarray:
        return dout


class Softmax:
    # multi-class output (Iris); converts logits to probabilities

    @staticmethod
    def forward(z: np.ndarray) -> np.ndarray:
        # subtract row max before exp for numerical stability
        shifted = z - np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(shifted)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    @staticmethod
    def backward(dout: np.ndarray, cache: np.ndarray) -> np.ndarray:
        """Jacobian-vector product for softmax.

        cache is the forward softmax probabilities. 
        When training with SoftmaxCrossEntropy, skip this — that loss already returns dL/dlogits.
        """
        sum_dout_s = np.sum(dout * cache, axis=-1, keepdims=True)
        return cache * (dout - sum_dout_s)


ACTIVATIONS = {
    "sigmoid": Sigmoid,
    "relu": ReLU,
    "tanh": Tanh,
    "linear": Linear,
    "softmax": Softmax,
}


def get_activation(name: str):

    key = name.lower()
    if key not in ACTIVATIONS:
        raise ValueError(f"Unknown activation {name!r}. Choose from {sorted(ACTIVATIONS)}")
    return ACTIVATIONS[key]
