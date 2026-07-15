from __future__ import annotations
import numpy as np

_EPS = 1e-12

class BinaryCrossEntropy:

    @staticmethod
    def forward(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        p = np.clip(y_pred, _EPS, 1.0 - _EPS)
        y = y_true.astype(float)
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    @staticmethod
    def backward(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        p = np.clip(y_pred, _EPS, 1.0 - _EPS)
        y = y_true.astype(float)
        n = y_pred.shape[0]
        return (-(y / p) + (1.0 - y) / (1.0 - p)) / n


class MSE:

    @staticmethod
    def forward(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return float(np.mean((y_pred - y_true.astype(float)) ** 2))

    @staticmethod
    def backward(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        n = y_pred.shape[0]
        return 2.0 * (y_pred - y_true.astype(float)) / n


class CategoricalCrossEntropy:
    @staticmethod
    def forward(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        p = np.clip(y_pred, _EPS, 1.0)
        if y_true.ndim == 1:
            n = y_true.shape[0]
            return float(-np.mean(np.log(p[np.arange(n), y_true.astype(int)])))
        return float(-np.mean(np.sum(y_true.astype(float) * np.log(p), axis=1)))

    @staticmethod
    def backward(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        p = np.clip(y_pred, _EPS, 1.0)
        n = y_pred.shape[0]
        if y_true.ndim == 1:
            grad = np.zeros_like(p)
            grad[np.arange(n), y_true.astype(int)] = -1.0 / p[np.arange(n), y_true.astype(int)]
            return grad / n
        return -(y_true.astype(float) / p) / n


class SoftmaxCrossEntropy:
    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_z = np.exp(shifted)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    @staticmethod
    def _one_hot(y_true: np.ndarray, num_classes: int) -> np.ndarray:
        if y_true.ndim == 2:
            return y_true.astype(float)
        n = y_true.shape[0]
        eye = np.zeros((n, num_classes), dtype=float)
        eye[np.arange(n), y_true.astype(int)] = 1.0
        return eye

    @staticmethod
    def forward(logits: np.ndarray, y_true: np.ndarray) -> float:
        probs = SoftmaxCrossEntropy._softmax(logits)
        return CategoricalCrossEntropy.forward(probs, y_true)

    @staticmethod
    def backward(logits: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        probs = SoftmaxCrossEntropy._softmax(logits)
        one_hot = SoftmaxCrossEntropy._one_hot(y_true, probs.shape[1])
        n = logits.shape[0]
        return (probs - one_hot) / n
