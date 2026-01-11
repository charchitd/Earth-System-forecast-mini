# Using ERA5 (optional)

This repo runs **offline** with a synthetic NetCDF dataset.

## If you already have NetCDF data
1) Put NetCDF in `data/` (e.g. `data/era5_t2m.nc`)
2) Update `configs/config.yaml`:
```yaml
dataset:
  path: data/era5_t2m.nc
  var: t2m
```
3) Ensure dims are `time, lat, lon` (or adapt name mapping in `src/data.py`).

## If you need to download ERA5
Use the Copernicus CDS API (requires an account + API key). Download a small region and time range first.
