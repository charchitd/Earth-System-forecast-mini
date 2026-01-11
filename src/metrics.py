from __future__ import annotations
import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Anomaly Correlation Coefficient averaged over samples."""
    yt = y_true.reshape(y_true.shape[0], -1)
    yp = y_pred.reshape(y_pred.shape[0], -1)

    yt = yt - yt.mean(axis=1, keepdims=True)
    yp = yp - yp.mean(axis=1, keepdims=True)

    num = np.sum(yt * yp, axis=1)
    den = np.sqrt(np.sum(yt ** 2, axis=1) * np.sum(yp ** 2, axis=1) + 1e-12)
    return float(np.mean(num / den))


def skill_score(baseline_rmse: float, model_rmse: float) -> float:
    return float(1.0 - (model_rmse / (baseline_rmse + 1e-12)))
