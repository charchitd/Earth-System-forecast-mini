from __future__ import annotations
import numpy as np
import xarray as xr


def make_lagged_samples(anom: xr.DataArray, n_lags: int, lead: int):
    """
    Convert (time, lat, lon) anomalies into supervised samples.
    X: (n_samples, n_lags, lat, lon)
    y: (n_samples, lat, lon) at t+lead
    """
    arr = anom.values  # (T, H, W)
    T, H, W = arr.shape
    start = n_lags - 1
    end = T - lead - 1
    n_samples = end - start + 1
    X = np.zeros((n_samples, n_lags, H, W), dtype=np.float32)
    y = np.zeros((n_samples, H, W), dtype=np.float32)

    for i, t in enumerate(range(start, end + 1)):
        X[i] = arr[t - n_lags + 1 : t + 1]
        y[i] = arr[t + lead]
    return X, y


def time_split(X, y, train_frac=0.7, val_frac=0.15):
    n = X.shape[0]
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
