"""
Time Machine — Lagrangian particle tracking for ocean current data.
Drops virtual particles and advects them to simulate plastic drift trajectories.
"""
import argparse
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from config import (
    get_u_path, get_v_path,
    U_VAR, V_VAR, LON_VAR, LAT_VAR, TIME_DIM, TIME_INDEX, ENGINE,
    NUM_PARTICLES, DAYS_TO_SIMULATE, START_LON, START_LAT_MIN, START_LAT_MAX,
    PLOT_XLIM, PLOT_YLIM,
)


def parse_args():
    p = argparse.ArgumentParser(description="Time Machine: Lagrangian particle tracking")
    p.add_argument("--u-file", default=None, help="Path to u-velocity NetCDF")
    p.add_argument("--v-file", default=None, help="Path to v-velocity NetCDF")
    p.add_argument("--u-var", default=U_VAR, help=f"U variable (default: {U_VAR})")
    p.add_argument("--v-var", default=V_VAR, help=f"V variable (default: {V_VAR})")
    p.add_argument("--lon-var", default=LON_VAR, help=f"Longitude var (default: {LON_VAR})")
    p.add_argument("--lat-var", default=LAT_VAR, help=f"Latitude var (default: {LAT_VAR})")
    p.add_argument("--time-dim", default=TIME_DIM, help=f"Time dimension (default: {TIME_DIM})")
    p.add_argument("--time-index", type=int, default=TIME_INDEX, help=f"Time slice (default: {TIME_INDEX})")
    p.add_argument("--engine", default=ENGINE, help=f"NetCDF engine (default: {ENGINE})")
    p.add_argument("--particles", type=int, default=NUM_PARTICLES, help=f"Number of particles (default: {NUM_PARTICLES})")
    p.add_argument("--days", type=int, default=DAYS_TO_SIMULATE, help=f"Simulation days (default: {DAYS_TO_SIMULATE})")
    p.add_argument("--start-lon", type=float, default=START_LON, help=f"Release longitude (default: {START_LON})")
    p.add_argument("--start-lat-min", type=float, default=START_LAT_MIN, help=f"Release lat min (default: {START_LAT_MIN})")
    p.add_argument("--start-lat-max", type=float, default=START_LAT_MAX, help=f"Release lat max (default: {START_LAT_MAX})")
    p.add_argument("--xlim", nargs=2, type=float, default=None, help="Plot x limits (e.g. 135 235)")
    p.add_argument("--ylim", nargs=2, type=float, default=None, help="Plot y limits (e.g. 20 48)")
    return p.parse_args()


def get_velocity_at_point(x, y, u_grid, v_grid, x_axis, y_axis):
    """Nearest-neighbor velocity lookup. Land (NaN) returns zero."""
    xi = (np.abs(x_axis[0, :] - x)).argmin()
    yi = (np.abs(y_axis[:, 0] - y)).argmin()
    u_val = u_grid[yi, xi]
    v_val = v_grid[yi, xi]
    if np.isnan(u_val): u_val = 0
    if np.isnan(v_val): v_val = 0
    return u_val, v_val


def main():
    args = parse_args()
    u_path = args.u_file or str(get_u_path())
    v_path = args.v_file or str(get_v_path())

    print("1. Loading Ocean Currents...")
    try:
        ds_u = xr.open_dataset(u_path, engine=args.engine)
        ds_v = xr.open_dataset(v_path, engine=args.engine)

        time_select = {args.time_dim: args.time_index} if args.time_dim in ds_u[args.u_var].dims else {}
        u_data = ds_u[args.u_var].isel(**time_select).squeeze().values
        v_data = ds_v[args.v_var].isel(**time_select).squeeze().values
        lons = getattr(ds_u, args.lon_var).values
        lats = getattr(ds_u, args.lat_var).values
        if lons.ndim == 1:
            lons, lats = np.meshgrid(lons, lats)
        print("   Data loaded successfully.")
    except Exception as e:
        print(f"   ERROR: {e}")
        return

    n = args.particles
    particles_x = np.full(n, args.start_lon)
    particles_y = np.linspace(args.start_lat_min, args.start_lat_max, n)
    print(f"2. Dropping {n} particles in a line...")

    history_x, history_y = [], []
    dt = 6 * 3600  # 6-hour steps
    meters_per_degree = 111000.0
    total_steps = args.days * 4

    print(f"3. Simulating {args.days} days...")
    for step in range(total_steps):
        for i in range(n):
            u_curr, v_curr = get_velocity_at_point(
                particles_x[i], particles_y[i], u_data, v_data, lons, lats
            )
            dx = u_curr * dt
            dy = v_curr * dt
            particles_x[i] += dx / (meters_per_degree * np.cos(np.radians(particles_y[i])))
            particles_y[i] += dy / meters_per_degree

        if step % 4 == 0:
            history_x.append(particles_x.copy())
            history_y.append(particles_y.copy())
            if (step / 4) % 30 == 0:
                print(f"   ... Day {int(step/4)} complete")

    print("4. Generating Animation...")
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.quiver(lons[::15, ::15], lats[::15, ::15],
              u_data[::15, ::15], v_data[::15, ::15],
              color='lightgray', alpha=0.5, scale=40, zorder=1)

    land_mask = np.isnan(u_data)
    ax.contourf(lons, lats, land_mask, levels=[0.5, 1.5], colors=['#D2B48C'], zorder=2)
    ax.contour(lons, lats, land_mask, levels=[0.5], colors='black', linewidths=0.5, zorder=3)

    scatter = ax.scatter([], [], c='red', s=25, edgecolor='black', zorder=5)
    title = ax.set_title("Day 0: The Launch")

    xlim = args.xlim or PLOT_XLIM or (lons.min(), lons.max())
    ylim = args.ylim or PLOT_YLIM or (lats.min(), lats.max())
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("Longitude (Degrees East)")
    ax.set_ylabel("Latitude")
    ax.set_facecolor('#E0F0FF')

    def update(frame):
        scatter.set_offsets(np.c_[history_x[frame], history_y[frame]])
        title.set_text(f"Day {frame}: Tracking the Debris")
        return scatter, title

    anim = FuncAnimation(fig, update, frames=len(history_x), interval=50, blit=False)
    print("Done! Check the pop-up window.")
    plt.show()


if __name__ == "__main__":
    main()
