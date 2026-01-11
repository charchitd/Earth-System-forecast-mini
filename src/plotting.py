from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt


def save_metrics(metrics: dict, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(metrics, indent=2))


def plot_skill(metrics: dict, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    labels, rmses = [], []
    for k, v in metrics["models"].items():
        labels.append(k)
        rmses.append(v["rmse"])

    plt.figure()
    plt.bar(labels, rmses)
    plt.ylabel("RMSE (anomaly units)")
    plt.title(f"Forecast comparison (lead={metrics['lead']} days)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
