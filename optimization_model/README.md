# Ocean plastic cleanup — optimization model

This folder contains everything needed to run the **main optimization notebook**:

- `main_optimization_model.ipynb` — drift simulation, density grid, MILP/greedy routing, validation plots
- `Aivan_EDA/` — `hycom_data.py`, `river_data.py`, and Meijer river data under `Aivan_EDA/data/`

## How to run

1. Create a Python environment and install dependencies from the repo root: `pip install -r requirements.txt`
2. Place your Gurobi license as `gurobi.lic` in the **repo root** or in this folder (the notebook searches both).
3. Open `main_optimization_model.ipynb` and run all cells.  
   You can run Jupyter with the working directory set to **this folder** or the **repo root**; path resolution finds `Aivan_EDA` automatically.

## Network

HYCOM and Meijer data may be fetched on first run (internet required).
