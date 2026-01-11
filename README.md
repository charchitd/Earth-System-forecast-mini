# Earth System Forecast Mini-Study (AI → Earth System Prediction)

A compact, reproducible mini-project that mirrors an **Earth-system prediction** workflow on gridded spatiotemporal data:
- load a (NetCDF) “reanalysis-like” dataset (synthetic by default),
- compute **anomalies**,
- run **time-based splits**,
- benchmark **baselines** (climatology, persistence, ridge),
- train a simple **ML model** (PyTorch MLP),
- report skill (RMSE / ACC) and save plots.

> Default data is **synthetic** (so the repo runs fully offline). You can swap in ERA5/other NetCDF later (instructions included).

## Quickstart
```bash
pip install -r requirements.txt
python scripts/run_experiment.py --lead 3
```

Outputs:
- `figures/metrics_lead3.json`
- `figures/skill_lead3.png`

## Data
- Offline default: `data/synthetic_reanalysis.nc` (generated automatically if missing)
- Optional: Use ERA5/other NetCDF. See `docs/ERA5.md`.

## Repo structure
```
.
├── data/
├── scripts/run_experiment.py
├── src/
│   ├── data.py
│   ├── features.py
│   ├── models.py
│   ├── metrics.py
│   └── plotting.py
├── figures/
├── configs/config.yaml
└── docs/ERA5.md
```

## Future work
- Replace synthetic data with ERA5 and add more variables (t2m, u10/v10, MSLP)
- Add spatial models (ConvNet / ConvLSTM) and probabilistic forecasts (ensembles, CRPS)
- Add S2S-style evaluation (week-3/4 skill, teleconnection indices)
- Add a toy data assimilation demo (Kalman filter on a gridded field)

## License
MIT
