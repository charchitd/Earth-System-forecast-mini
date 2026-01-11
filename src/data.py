from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr


@dataclass
class DatasetConfig:
    path: str
    var: str = "t2m"


def generate_synthetic_reanalysis(path: str, seed: int = 42) -> None:
    """Generate a small reanalysis-like NetCDF dataset (daily, lat-lon grid)."""
    rng = np.random.default_rng(seed)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_days = 365 * 4  # 4 years daily
    time = pd.date_range("2000-01-01", periods=n_days, freq="D")
    lat = np.linspace(-40, 40, 16).astype(np.float32)
    lon = np.linspace(0, 360 - 360/32, 32).astype(np.float32)

    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    spatial = 2.0 * np.sin(np.deg2rad(lon2d)) + 1.5 * np.cos(np.deg2rad(lat2d))
    lat_grad = -0.08 * np.abs(lat2d)

    t = np.arange(n_days, dtype=np.float32)
    seasonal = 8.0 * np.sin(2 * np.pi * t / 365.0) + 2.0 * np.cos(2 * np.pi * t / 180.0)

    ar = np.zeros((n_days, lat.size, lon.size), dtype=np.float32)
    eps = rng.normal(0, 1.0, size=ar.shape).astype(np.float32)
    phi = 0.85
    for i in range(1, n_days):
        ar[i] = phi * ar[i - 1] + eps[i]

    base = 285.0
    field = base + seasonal[:, None, None] + spatial[None, :, :] + lat_grad[None, :, :] + 0.8 * ar

    ds = xr.Dataset(
        {"t2m": (("time", "lat", "lon"), field.astype(np.float32))},
        coords={"time": time, "lat": lat, "lon": lon},
        attrs={"description": "Synthetic reanalysis-like dataset for forecasting demos."},
    )
    ds.to_netcdf(out)


def load_dataset(cfg: DatasetConfig) -> xr.Dataset:
    path = Path(cfg.path)
    if not path.exists():
        generate_synthetic_reanalysis(cfg.path)
    ds = xr.open_dataset(path)

    # Normalize coordinate names if needed
    if "latitude" in ds.coords and "lat" not in ds.coords:
        ds = ds.rename({"latitude": "lat"})
    if "longitude" in ds.coords and "lon" not in ds.coords:
        ds = ds.rename({"longitude": "lon"})

    if cfg.var not in ds:
        raise KeyError(f"Variable '{cfg.var}' not found. Available: {list(ds.data_vars)}")
    return ds


def anomalies_daily(ds: xr.Dataset, var: str) -> xr.DataArray:
    """Remove day-of-year climatology to compute anomalies."""
    da = ds[var]
    clim = da.groupby("time.dayofyear").mean("time")
    anom = da.groupby("time.dayofyear") - clim
    return anom.astype(np.float32)
