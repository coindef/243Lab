"""
Configuration for Plastic Hunter EDA scripts.
Override via environment variables or argparse for different datasets.
"""
import os
from pathlib import Path

# --- Paths (relative to DATA_DIR or absolute) ---
_script_dir = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("PLASTIC_HUNTER_DATA", _script_dir))
U_FILE = os.environ.get("U_VEL_FILE", "2018_uvel.nc4")
V_FILE = os.environ.get("V_VEL_FILE", "2018_vvel.nc4")

# --- NetCDF variable names (varies by dataset: HYCOM, Copernicus, etc.) ---
U_VAR = os.environ.get("U_VAR", "u")           # e.g. "u", "uo"
V_VAR = os.environ.get("V_VAR", "v")           # e.g. "v", "vo"
LON_VAR = os.environ.get("LON_VAR", "Longitude")  # e.g. "Longitude", "longitude", "lon"
LAT_VAR = os.environ.get("LAT_VAR", "Latitude")   # e.g. "Latitude", "latitude", "lat"
TIME_DIM = os.environ.get("TIME_DIM", "MT")    # e.g. "MT", "time", "time_counter"
TIME_INDEX = int(os.environ.get("TIME_INDEX", "0"))

# --- NetCDF engine ---
ENGINE = os.environ.get("NETCDF_ENGINE", "h5netcdf")  # "h5netcdf" or "netcdf4"

# --- Simulation defaults ---
NUM_PARTICLES = int(os.environ.get("NUM_PARTICLES", "150"))
DAYS_TO_SIMULATE = int(os.environ.get("DAYS_TO_SIMULATE", "180"))
START_LON = float(os.environ.get("START_LON", "145.0"))
START_LAT_MIN = float(os.environ.get("START_LAT_MIN", "30.0"))
START_LAT_MAX = float(os.environ.get("START_LAT_MAX", "40.0"))

# --- Plot bounds (auto from data if None) ---
PLOT_XLIM = None  # e.g. (135, 235)
PLOT_YLIM = None  # e.g. (20, 48)


def get_u_path():
    p = Path(U_FILE)
    return p if p.is_absolute() else DATA_DIR / p


def get_v_path():
    p = Path(V_FILE)
    return p if p.is_absolute() else DATA_DIR / p
