from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

from sklearn.linear_model import Ridge

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainConfig:
    epochs: int = 25
    batch_size: int = 16
    lr: float = 1e-3
    hidden: int = 256
    seed: int = 42


def baseline_climatology(X: np.ndarray) -> np.ndarray:
    n, _, H, W = X.shape
    return np.zeros((n, H, W), dtype=np.float32)


def baseline_persistence(X: np.ndarray) -> np.ndarray:
    return X[:, -1, :, :].astype(np.float32)


def ridge_regression(X_train, y_train, X_eval, alpha: float = 1.0) -> np.ndarray:
    n_train = X_train.shape[0]
    n_eval = X_eval.shape[0]
    Xtr = X_train.reshape(n_train, -1)
    ytr = y_train.reshape(n_train, -1)
    Xev = X_eval.reshape(n_eval, -1)

    model = Ridge(alpha=alpha, random_state=0)
    model.fit(Xtr, ytr)
    yhat = model.predict(Xev).reshape(n_eval, y_train.shape[1], y_train.shape[2]).astype(np.float32)
    return yhat


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(X_train, y_train, X_val, y_val, cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_dim = int(np.prod(X_train.shape[1:]))
    out_dim = int(np.prod(y_train.shape[1:]))

    def to_tensors(X, y):
        Xt = torch.from_numpy(X.reshape(X.shape[0], -1)).float()
        yt = torch.from_numpy(y.reshape(y.shape[0], -1)).float()
        return Xt, yt

    Xtr, ytr = to_tensors(X_train, y_train)
    Xva, yva = to_tensors(X_val, y_val)

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xva, yva), batch_size=cfg.batch_size, shuffle=False)

    model = MLP(in_dim=in_dim, out_dim=out_dim, hidden=cfg.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None

    for _ in range(cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_losses.append(loss_fn(pred, yb).item())

        v = float(np.mean(val_losses))
        if v < best_val:
            best_val = v
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {"best_val_mse": best_val, "device": str(device)}


def predict_mlp(model: nn.Module, X: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    Xt = torch.from_numpy(X.reshape(X.shape[0], -1)).float().to(device)
    model.eval()
    with torch.no_grad():
        yp = model(Xt).cpu().numpy()
    H, W = X.shape[2], X.shape[3]
    return yp.reshape(X.shape[0], H, W).astype(np.float32)
