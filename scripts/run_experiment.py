from __future__ import annotations
import argparse
from pathlib import Path
import yaml

from src.data import DatasetConfig, load_dataset, anomalies_daily
from src.features import make_lagged_samples, time_split
from src.models import (
    TrainConfig,
    baseline_climatology,
    baseline_persistence,
    ridge_regression,
    train_mlp,
    predict_mlp,
)
from src.metrics import rmse, acc, skill_score
from src.plotting import save_metrics, plot_skill


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--lead", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    lead = args.lead if args.lead is not None else int(cfg["forecast"]["lead"])

    ds_cfg = DatasetConfig(**cfg["dataset"])
    ds = load_dataset(ds_cfg)
    anom = anomalies_daily(ds, ds_cfg.var)

    X, y = make_lagged_samples(anom, n_lags=int(cfg["features"]["n_lags"]), lead=lead)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = time_split(
        X, y,
        train_frac=float(cfg["split"]["train_frac"]),
        val_frac=float(cfg["split"]["val_frac"]),
    )

    # Baselines
    yhat_clim = baseline_climatology(Xte)
    yhat_pers = baseline_persistence(Xte)

    # Ridge
    yhat_ridge = ridge_regression(Xtr, ytr, Xte, alpha=1.0)

    # MLP
    tcfg = TrainConfig(**cfg["training"])
    mlp, train_info = train_mlp(Xtr, ytr, Xva, yva, tcfg)
    yhat_mlp = predict_mlp(mlp, Xte)

    # Metrics
    rmse_pers = rmse(yte, yhat_pers)
    models = {
        "climatology": {"rmse": rmse(yte, yhat_clim), "acc": acc(yte, yhat_clim)},
        "persistence": {"rmse": rmse_pers, "acc": acc(yte, yhat_pers)},
        "ridge": {"rmse": rmse(yte, yhat_ridge), "acc": acc(yte, yhat_ridge)},
        "mlp": {"rmse": rmse(yte, yhat_mlp), "acc": acc(yte, yhat_mlp), **train_info},
    }

    for name in models:
        models[name]["skill_vs_persistence"] = 0.0 if name == "persistence" else skill_score(rmse_pers, models[name]["rmse"])

    out = {
        "lead": lead,
        "n_samples": int(X.shape[0]),
        "test_samples": int(Xte.shape[0]),
        "models": models,
    }

    metrics_path = f"figures/metrics_lead{lead}.json"
    fig_path = f"figures/skill_lead{lead}.png"
    save_metrics(out, metrics_path)
    plot_skill(out, fig_path)

    print(f"Saved metrics → {metrics_path}")
    print(f"Saved figure  → {fig_path}")


if __name__ == "__main__":
    main()
