"""
Ocean Scanner — Physics-aware EDA for ocean current data.
Produces speed heatmap and vector field to validate flow direction and identify features.
"""
import argparse
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

from config import (
    get_u_path, get_v_path,
    U_VAR, V_VAR, LON_VAR, LAT_VAR, TIME_DIM, TIME_INDEX, ENGINE,
)

def parse_args():
    p = argparse.ArgumentParser(description="Ocean Scanner: EDA for ocean current NetCDF data")
    p.add_argument("--u-file", default=None, help="Path to u-velocity NetCDF (default: config)")
    p.add_argument("--v-file", default=None, help="Path to v-velocity NetCDF (default: config)")
    p.add_argument("--u-var", default=U_VAR, help=f"U variable name (default: {U_VAR})")
    p.add_argument("--v-var", default=V_VAR, help=f"V variable name (default: {V_VAR})")
    p.add_argument("--lon-var", default=LON_VAR, help=f"Longitude variable (default: {LON_VAR})")
    p.add_argument("--lat-var", default=LAT_VAR, help=f"Latitude variable (default: {LAT_VAR})")
    p.add_argument("--time-dim", default=TIME_DIM, help=f"Time dimension (default: {TIME_DIM})")
    p.add_argument("--time-index", type=int, default=TIME_INDEX, help=f"Time slice index (default: {TIME_INDEX})")
    p.add_argument("--engine", default=ENGINE, help=f"NetCDF engine (default: {ENGINE})")
    return p.parse_args()


def main():
    args = parse_args()
    u_path = args.u_file or str(get_u_path())
    v_path = args.v_file or str(get_v_path())

    print("1. Loading data...")
    ds_u = xr.open_dataset(u_path, engine=args.engine)
    ds_v = xr.open_dataset(v_path, engine=args.engine)

    u = ds_u[args.u_var].squeeze()
    v = ds_v[args.v_var].squeeze()

    print("2. Calculating current speeds...")
    speed = np.sqrt(u**2 + v**2)

    # Time slice (handle datasets with/without time dim)
    time_select = {args.time_dim: args.time_index} if args.time_dim in u.dims else {}
    u_slice = u.isel(**time_select) if time_select else u
    v_slice = v.isel(**time_select) if time_select else v
    speed_slice = speed.isel(**time_select) if time_select else speed

    X = getattr(ds_u, args.lon_var)
    Y = getattr(ds_u, args.lat_var)

    print("3. Generating plot...")
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))

    # --- PLOT 1: Speed Heatmap ---
    speed_slice.plot(ax=ax[0], cmap='viridis', vmin=0, vmax=1.5, cbar_kwargs={'label': 'Current Speed (m/s)'})
    ax[0].set_title("Current Speed (Blue = Plastic Traps)")

    # --- PLOT 2: Vector Field ---
    step = 10
    X_vals = X.values
    Y_vals = Y.values
    if X_vals.ndim == 1:
        X_grid, Y_grid = np.meshgrid(X_vals, Y_vals)
    else:
        X_grid, Y_grid = X_vals, Y_vals
    X_sub = X_grid[::step, ::step]
    Y_sub = Y_grid[::step, ::step]
    U_sub = u_slice.values[::step, ::step] if u_slice.ndim >= 2 else u_slice.values[::step]
    V_sub = v_slice.values[::step, ::step] if v_slice.ndim >= 2 else v_slice.values[::step]

    ax[1].quiver(X_sub, Y_sub, U_sub, V_sub, scale=20, width=0.002)
    ax[1].set_title("Surface Currents (Look for the Gyre)")
    ax[1].set_xlabel("Longitude")
    ax[1].set_ylabel("Latitude")

    print("Done! Check the window.")
    plt.show()


if __name__ == "__main__":
    main()
