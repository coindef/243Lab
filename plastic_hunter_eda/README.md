# Plastic Hunter Analytics Suite

The **Intelligence Layer** of a Dynamic Ocean Remediation System for the North Pacific Gyre. Exploratory data analysis (EDA) and Lagrangian particle tracking for ocean current NetCDF data—generalized to work with HYCOM, Copernicus, and other datasets.

---

## What It Does

### 1. Ocean Scanner (`EDA.py`)

Physics-aware EDA: speed heatmap + vector field to validate flow direction and identify features (e.g., Kuroshio Extension, North Pacific Gyre).

### 2. Time Machine (`particle_simulation.py`)

Lagrangian particle tracking: drops virtual particles, advects them with current velocities, and animates plastic drift trajectories over time.

---

## Requirements

```bash
pip install -r requirements.txt
```

- xarray, numpy, matplotlib, h5netcdf

---

## Data

Expects ocean current NetCDF files: u-velocity and v-velocity (m/s). Land cells as NaN. Compatible with HYCOM, Copernicus Marine Service, etc.

---

## Usage

### Default (HYCOM-style: `2018_uvel.nc4`, `2018_vvel.nc4` in script directory)

```bash
cd plastic_hunter_eda
python EDA.py
python particle_simulation.py
```

### Custom paths and variable names

```bash
# Different file paths
python EDA.py --u-file /data/uo.nc --v-file /data/vo.nc

# Copernicus variable names (uo, vo, longitude, latitude, time)
python EDA.py --u-file uo.nc --v-file vo.nc --u-var uo --v-var vo --lon-var longitude --lat-var latitude --time-dim time

# Particle simulation: custom release zone and plot bounds
python particle_simulation.py --u-file uo.nc --v-file vo.nc --start-lon 150 --start-lat-min 25 --start-lat-max 45 --xlim 120 260 --ylim 15 50
```

### Environment variables

| Variable | Purpose |
|----------|---------|
| `PLASTIC_HUNTER_DATA` | Data directory (default: script dir) |
| `U_VEL_FILE`, `V_VEL_FILE` | Default filenames |
| `U_VAR`, `V_VAR` | Velocity variable names |
| `LON_VAR`, `LAT_VAR` | Coordinate variable names |
| `TIME_DIM` | Time dimension name |
| `TIME_INDEX` | Time slice index |

---

## File Structure

```
plastic_hunter_eda/
├── README.md
├── requirements.txt
├── config.py              # Central config (paths, vars); override via env or CLI
├── EDA.py                 # Ocean Scanner
└── particle_simulation.py # Time Machine
```

Large `.nc4` / `.nc` files are excluded via `.gitignore`.
