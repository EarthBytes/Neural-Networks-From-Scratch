from __future__ import annotations
import numpy as np

class Sigmoid:

    @staticmethod
    def forward(z: np.ndarray) -> np.ndarray:
        # Stable form: avoid overflow for large |z|
        out = np.empty_like(z, dtype=float)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        exp_z = np.exp(z[~pos])
        out[~pos] = exp_z / (1.0 + exp_z)
        return out

    @staticmethod
    def backward(dout: np.ndarray, cache: np.ndarray) -> np.ndarray:
        return dout * cache * (1.0 - cache)


class ReLU:
    
    @staticmethod
    def forward(z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    @staticmethod
    def backward(dout: np.ndarray, cache: np.ndarray) -> np.ndarray:
        dz = dout.copy()
        dz[cache <= 0] = 0.0
        return dz


class Tanh:
   
    @staticmethod
    def forward(z: np.ndarray) -> np.ndarray:
        return np.tanh(z)

    @staticmethod
    def backward(dout: np.ndarray, cache: np.ndarray) -> np.ndarray:
        return dout * (1.0 - cache**2)


class Linear:

    @staticmethod
    def forward(z: np.ndarray) -> np.ndarray:
        return z

    @staticmethod
    def backward(dout: np.ndarray, cache: np.ndarray) -> np.ndarray:
        return dout


class Softmax:

    @staticmethod
    def forward(z: np.ndarray) -> np.ndarray:
        shifted = z - np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(shifted)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    @staticmethod
    def backward(dout: np.ndarray, cache: np.ndarray) -> np.ndarray:
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
