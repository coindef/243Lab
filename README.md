# 243Lab — Plastic Hunter

“Plastic Hunter” models ocean cleanup as a dynamic pursuit problem: predict drifting plastic patches and route vessels to maximize collection per fuel cost.

## Repository layout

| Path | Purpose |
|------|---------|
| **`optimization_model/`** | **Main workflow:** `main_optimization_model.ipynb` plus `Aivan_EDA/` (HYCOM + Meijer river data loaders). This is the self-contained optimization stack. |
| `eda_notebooks/` | Exploratory notebooks (HYCOM EDA, Meijer EDA, older Step2 variants, simulation v3). |
| `docs/` | Documentation (e.g. model evaluation notes). |
| `legacy_plastic_hunter/` | Older `plastic_hunter_eda` code (not required for the current optimization notebook). |
| `legacy_duplicate/` | Nested/duplicate copies of datasets and an old nested `243Lab` tree (kept for reference; safe to delete after review). |
| `requirements.txt` | Python dependencies |
| `LICENSE` | License |

### Quick start (optimization)

See **`optimization_model/README.md`**.

### Data sources

- HYCOM Global Ocean Prediction System (currents)
- Meijer et al. 2021 river plastic emissions (via `optimization_model/Aivan_EDA/river_data.py`)
