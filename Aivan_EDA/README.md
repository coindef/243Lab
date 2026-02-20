# Plastic Hunter Analytics Suite

The **Intelligence Layer** of a Dynamic Ocean Remediation System for the North Pacific Gyre. This repository contains exploratory data analysis (EDA) and Lagrangian particle tracking tools that transform raw oceanographic current data into kinetic energy maps and simulated plastic drift trajectories.

---

## What It Does

### 1. Ocean Scanner (`EDA.py`)

Before cleaning the ocean, you must understand the battlefield. This script:

- **Loads** raw ocean current data (NetCDF format: u-velocity and v-velocity)
- **Computes** current speed as kinetic energy: `speed = √(u² + v²)`
- **Visualizes** two critical views:
  - **Speed heatmap** — Identifies the "Highway" (Kuroshio Extension) and the "Trap" (North Pacific Gyre)
  - **Vector field** — Confirms flow direction (clockwise gyre) and coordinate integrity (Japan to Hawaii)

This is physics-aware EDA: it validates that your data matches real ocean behavior, not just missing values.

---

### 2. Time Machine (`particle_simulation.py`)

Static maps lie. This script is a **Lagrangian particle tracking model** that shows where water—and plastic—actually goes:

- **Drops** 150 virtual particles along a vertical line east of Japan (latitudes 30°N–40°N)
- **Advects** them using 6-hour time steps over 180 days
- **Interpolates** velocities from the current grid (nearest-neighbor at each step)
- **Animates** the results with land overlay and coastlines

**Key insight:** Only particles entering at certain latitudes (e.g., 34–35°N) reach the gyre. Trash at 40°N gets caught in coastal eddies. This "selection bias" has direct policy implications for where to prioritize river filtration.

---

## Requirements

- **Python 3.8+**

Install dependencies:

```bash
pip install -r requirements.txt
```

| Package     | Purpose                          |
|------------|-----------------------------------|
| xarray     | NetCDF I/O and labeled arrays     |
| numpy      | Numerical operations              |
| matplotlib | Plots and animation               |
| h5netcdf   | NetCDF4 engine for `.nc4` files   |
| netcdf4    | OPeNDAP access (remote data)      |
| pandas     | Time handling for multi-year EDA  |

---

## Data: Local Files OR OPeNDAP (No Download)

**Option A — OPeNDAP (recommended):** No data download needed. Data streams from HYCOM's THREDDS server. No API key required.

- **Newest data:** ESPC-D-V02 (Aug 2024 – Present)
- **Multi-year:** GOFS 3.1 Reanalysis (1994–2015) for seasonality and trend analysis

**Option B — Local files:** Place `2018_uvel.nc4` and `2018_vvel.nc4` in the project root (same format as before).

---

## Usage

### 1. Ocean Scanner (EDA)

```bash
python EDA.py
```

Uses local files if present; otherwise falls back to OPeNDAP.

### 2. Particle Simulation (Fast: ~2–3 min)

Uses **river plastic emissions** (Meijer 2021) for starting positions and **2024** ocean data.

```bash
python particle_simulation.py
```

- **River sources**: 10,000+ Asia-Pacific rivers, emission-weighted ([Meijer et al. 2021](https://figshare.com/articles/dataset/Supplementary_data_for_More_than_1000_rivers_account_for_80_of_global_riverine_plsatic_emissions_into_the_ocean_/14515590))
- `"opendap_analysis"` — 2024 data (Dec 2018–Sep 2024), **default**
- `"opendap_reanalysis"` — 1994–2015 (time-varying, ~30 min)
- `"local"` — Your `2018_uvel.nc4` / `2018_vvel.nc4` files

### 3. Trend Analysis (2019–2024 Averages, ~5–10 min)

```bash
python EDA_trends_recent.py
```

Loads 1 month per year, computes mean speed, plots trends.

### 4. Multi-Year EDA (Seasonality & Trends, 1994–2015)

```bash
python EDA_multi_year.py
```

Loads reanalysis data in yearly chunks and produces:

- **Seasonal comparison** — Winter vs summer gyre strength
- **Year-over-year trend** — Mean current speed by year
- **Seasonal heatmap** — Speed by year and month
- **Spatial map** — Mean July speed in the gyre region

---

## Project Context

This forms **Module 1** of a larger system. The conclusion:

> *"Static cleanup arrays fail because the trap moves. The only viable solution is a dynamic routing algorithm."*

**Module 2** will use these trajectory forecasts to optimize boat routing—"skating to where the puck is going to be"—for plastic interception.

---

## File Structure

```
analytics-lab/
├── README.md
├── requirements.txt
├── hycom_data.py           # OPeNDAP data fetcher (no API key)
├── river_data.py           # River plastic emissions (Meijer 2021)
├── EDA.py                  # Ocean Scanner: speed map + vector field
├── EDA_trends_recent.py    # Trend analysis (2019–2024, ~5–10 min)
├── EDA_multi_year.py       # Seasonality & trend analysis (1994–2015)
├── particle_simulation.py  # Time Machine: Lagrangian particle tracking
├── .gitignore              # Excludes *.nc4 (large datasets)
└── 2018_uvel.nc4           # (local only, optional)
    2018_vvel.nc4           # (local only, optional)
```
