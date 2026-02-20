"""
HYCOM Data Fetcher - OPeNDAP API Access

Fetches ocean current data from HYCOM's THREDDS server without downloading files.
No API key required. Data streams directly to your Python session.

Data Sources:
  - analysis:  GOFS 3.1 Analysis (Dec 2018 - Sep 2024) - RECENT, for trends
  - newest:   ESPC-D-V02 (Aug 2024 - Present) - latest
  - reanalysis: GOFS 3.1 Reanalysis (1994-2015) - 22 years for long-term trends
"""

import xarray as xr
import numpy as np
from typing import Optional, Tuple

# OPeNDAP base URL (no API key needed - this IS the API)
THREDDS_BASE = "https://tds.hycom.org/thredds/dodsC"

# Approximate steps per year (3-hourly = 8/day, 365*8 = 2920)
STEPS_PER_YEAR = 2920

# Dataset configurations
DATASETS = {
    "analysis": {
        "name": "GOFS 3.1 Analysis (Dec 2018 - Sep 2024)",
        "url": f"{THREDDS_BASE}/GLBy0.08/expt_93.0",
        "u_var": "water_u",
        "v_var": "water_v",
        "lat_var": "lat",
        "lon_var": "lon",
        "time_var": "time",
        "depth_var": "depth",
        "surface_depth_idx": 0,
        "total_steps": 16809,
        "start_year": 2018,
    },
    "newest": {
        "name": "ESPC-D-V02 (Aug 2024 - Present)",
        "url": f"{THREDDS_BASE}/FMRC_ESPC-D-V02_uv3z/FMRC_ESPC-D-V02_uv3z_best.ncd",
        "u_var": "water_u",
        "v_var": "water_v",
        "lat_var": "lat",
        "lon_var": "lon",
        "time_var": "time",
        "depth_var": "depth",
        "surface_depth_idx": 0,
    },
    "reanalysis": {
        "name": "GOFS 3.1 Reanalysis (1994-2015)",
        "url": f"{THREDDS_BASE}/GLBv0.08/expt_53.X",
        "u_var": "water_u",
        "v_var": "water_v",
        "lat_var": "lat",
        "lon_var": "lon",
        "time_var": "time",
        "depth_var": "depth",
        "surface_depth_idx": 0,
    },
}


def get_step_range_for_year(source: str, year: int) -> Tuple[int, int]:
    """Get (step_start, step_end) for a given year. Analysis: Dec 2018 - Sep 2024."""
    cfg = DATASETS.get("analysis", {})
    total = cfg.get("total_steps", 16809)
    # Analysis starts Dec 4 2018: 2018 has ~27 days, 2019+ full years
    if year <= 2018:
        return 0, min(216, total)
    step_start = 216 + (year - 2019) * STEPS_PER_YEAR  # 2019 = 216, 2020 = 3136, ...
    step_end = min(step_start + STEPS_PER_YEAR, total)
    return step_start, step_end


def load_hycom_opendap(
    source: str = "reanalysis",
    time_slice: Optional[slice] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    decode_times: bool = True,
) -> xr.Dataset:
    """
    Load HYCOM data via OPeNDAP (streams from server, no local download).

    Parameters
    ----------
    source : str
        "newest" = ESPC-D-V02 (Aug 2024+), "reanalysis" = GOFS 3.1 (1994-2015)
    time_slice : slice, optional
        e.g. slice(0, 100) for first 100 time steps, or slice("2020-01-01", "2020-12-31")
    bbox : tuple, optional
        (lon_min, lon_max, lat_min, lat_max) to subset region (reduces data transfer)
    decode_times : bool
        Decode time to datetime64 (default True)

    Returns
    -------
    xarray.Dataset with u, v, lat, lon, time
    """
    cfg = DATASETS.get(source, DATASETS["reanalysis"])
    print(f"   Connecting to: {cfg['name']}")
    print(f"   URL: {cfg['url']}")

    # HYCOM reanalysis has a 'tau' variable with non-standard "hours since analysis"
    # which breaks decode_times. Open with decode_times=False, then manually decode.
    ds = xr.open_dataset(
        cfg["url"],
        engine="netcdf4",  # OPeNDAP uses netcdf4 engine
        decode_cf=True,
        decode_times=False,  # Avoid tau decoding errors
    )

    # Manually decode main time coordinate if requested
    if decode_times and "time" in ds.coords:
        import pandas as pd
        t = ds["time"].values
        # Common HYCOM units: "hours since 2000-01-01" or similar
        units = ds["time"].attrs.get("units", "hours since 2000-01-01")
        if "2000-01-01" in units:
            ds["time"] = pd.to_datetime("2000-01-01") + pd.to_timedelta(t, unit="h")
        elif "since" in units:
            # Parse "hours since YYYY-MM-DD"
            for part in units.replace(",", " ").split():
                if len(part) == 10 and part[4] == "-":
                    ds["time"] = pd.to_datetime(part) + pd.to_timedelta(t, unit="h")
                    break

    # Rename to common interface
    ds = ds.rename({
        cfg["u_var"]: "u",
        cfg["v_var"]: "v",
        cfg["lat_var"]: "lat",
        cfg["lon_var"]: "lon",
        cfg["time_var"]: "time",
    })

    # Surface layer only (depth index 0)
    if "depth" in ds.dims:
        ds = ds.isel(depth=cfg["surface_depth_idx"])

    if time_slice is not None:
        ds = ds.isel(time=time_slice)

    if bbox is not None:
        lon_min, lon_max, lat_min, lat_max = bbox
        ds = ds.sel(
            lon=slice(lon_min, lon_max),
            lat=slice(lat_min, lat_max),
        )

    return ds


def get_surface_velocity_arrays(
    ds: xr.Dataset,
    time_idx: int = 0,
) -> tuple:
    """
    Extract u, v, lons, lats as numpy arrays for particle simulation.
    Handles both 1D and 2D lat/lon grids.
    """
    u = ds["u"].isel(time=time_idx).squeeze().values
    v = ds["v"].isel(time=time_idx).squeeze().values
    lats = ds["lat"].values
    lons = ds["lon"].values

    # HYCOM reanalysis uses 1D lat/lon - create 2D mesh for interpolation
    if lons.ndim == 1 and lats.ndim == 1:
        lon_2d, lat_2d = np.meshgrid(lons, lats)
    else:
        lon_2d = lons
        lat_2d = lats

    return u, v, lon_2d, lat_2d


def get_time_varying_velocity(
    ds: xr.Dataset,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> tuple:
    """
    Get full time series of u, v for time-varying simulation.
    Returns: u_all, v_all, lons, lats, time_values
    """
    if bbox:
        lon_min, lon_max, lat_min, lat_max = bbox
        ds = ds.sel(
            lon=slice(lon_min, lon_max),
            lat=slice(lat_min, lat_max),
        )

    u_all = ds["u"].values  # (time, lat, lon)
    v_all = ds["v"].values
    lats = ds["lat"].values
    lons = ds["lon"].values
    times = ds["time"].values

    if lons.ndim == 1:
        lon_2d, lat_2d = np.meshgrid(lons, lats)
    else:
        lon_2d = lons
        lat_2d = lats

    return u_all, v_all, lon_2d, lat_2d, times


def load_velocity_chunk(
    source: str,
    step_start: int,
    step_end: int,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    stride: int = 2,
) -> tuple:
    """
    Load a chunk of velocity data (avoids OPeNDAP timeout on large requests).
    stride=2 loads every 2nd grid point (4x less data).
    Returns: u_chunk, v_chunk, lons, lats
    """
    ds = load_hycom_opendap(
        source=source,
        time_slice=slice(step_start, step_end),
        bbox=bbox,
    )
    # Subsample spatially to reduce transfer (stride=2 → 4x less data)
    if stride > 1:
        ds = ds.isel(lon=slice(None, None, stride), lat=slice(None, None, stride))
    u_chunk = ds["u"].values
    v_chunk = ds["v"].values
    lats = ds["lat"].values
    lons = ds["lon"].values
    if lons.ndim == 1:
        lon_2d, lat_2d = np.meshgrid(lons, lats)
    else:
        lon_2d, lat_2d = lons, lats
    try:
        ds.close()
    except Exception:
        pass
    return u_chunk, v_chunk, lon_2d, lat_2d
