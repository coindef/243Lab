"""
Lagrangian Particle Tracking - OPeNDAP support

Data sources:
  - opendap_analysis: 2024 data (Dec 2018 - Sep 2024) - RECENT, recommended
  - opendap_reanalysis: 1994-2015 (time-varying, slow ~30 min)
  - opendap_newest: Aug 2024+ (limited history)
  - local: 2018_uvel.nc4 / 2018_vvel.nc4
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. CONFIGURATION ---
NUM_PARTICLES = 150
DAYS_TO_SIMULATE = 180
START_LON = 145.0
START_LAT_MIN = 30.0
START_LAT_MAX = 40.0

# Use river emissions (Meijer 2021) for starting positions
USE_RIVER_SOURCES = True

# Data source: "opendap_analysis" | "opendap_reanalysis" | "opendap_newest" | "local"
DATA_SOURCE = "opendap_analysis"

# QUICK_MODE: 1 snapshot (~2-3 min). False = time-varying (~30 min, can timeout)
QUICK_MODE = True

# Save animation to GIF for slides (requires: pip install pillow)
SAVE_GIF = True
GIF_FILENAME = "particle_simulation.gif"

# North Pacific bounding box (ocean currents) - extend west to 110°E for full China coast
BBOX = (110, 220, 15, 50)
# River region: Asia-Pacific rivers feeding the gyre
RIVER_BBOX = (100, 180, 0, 55)

# For time-varying mode only (when QUICK_MODE=False)
CHUNK_DAYS = 7
CHUNK_STEPS = CHUNK_DAYS * 8
OPENDAP_STRIDE = 2

print("1. Loading Ocean Currents...")
try:
    if DATA_SOURCE.startswith("opendap"):
        from hycom_data import (
            load_hycom_opendap,
            get_surface_velocity_arrays,
            load_velocity_chunk,
            get_step_range_for_year,
        )

        if "analysis" in DATA_SOURCE:
            source = "analysis"
            # Use 2024 data: step ~16000 = mid-2024
            step_2024 = 16000
        elif "newest" in DATA_SOURCE:
            source = "newest"
            step_2024 = 0
        else:
            source = "reanalysis"
            step_2024 = 0

        use_time_varying = (
            not QUICK_MODE
            and source in ("reanalysis", "analysis")
        )

        if use_time_varying:
            chunk_start = step_2024 if source == "analysis" else 0
            print(f"   Using time-varying currents (chunked: {CHUNK_DAYS} days at a time)")
            try:
                u_all, v_all, lons, lats = load_velocity_chunk(
                    source, chunk_start, chunk_start + CHUNK_STEPS,
                    bbox=BBOX, stride=OPENDAP_STRIDE
                )
                u_data = v_data = None  # Not used in time-varying mode
                time_varying = True
                u_vis = u_all[0].copy()  # For final plot
                v_vis = v_all[0].copy()
            except Exception as chunk_err:
                print(f"   Chunked load failed: {chunk_err}")
                print("   Falling back to single snapshot...")
                ts = slice(step_2024, step_2024 + 1) if source == "analysis" else slice(0, 1)
                ds = load_hycom_opendap(source=source, time_slice=ts, bbox=BBOX)
                u_data, v_data, lons, lats = get_surface_velocity_arrays(ds, time_idx=0)
                time_varying = False
                u_vis = u_data
                v_vis = v_data
        else:
            ts = slice(step_2024, step_2024 + 1) if source == "analysis" else slice(0, 1)
            ds = load_hycom_opendap(source=source, time_slice=ts, bbox=BBOX)
            u_data, v_data, lons, lats = get_surface_velocity_arrays(ds, time_idx=0)
            time_varying = False
            u_vis, v_vis = u_data, v_data
            yr = "2024" if source == "analysis" else "latest"
            print(f"   Using single snapshot ({yr})")

    else:
        # Local files (original behavior)
        ds_u = xr.open_dataset("2018_uvel.nc4", engine="h5netcdf")
        ds_v = xr.open_dataset("2018_vvel.nc4", engine="h5netcdf")
        u_data = ds_u["u"].isel(MT=0).squeeze().values
        v_data = ds_v["v"].isel(MT=0).squeeze().values
        lons = ds_u.Longitude.values
        lats = ds_u.Latitude.values
        time_varying = False
        print("   Data loaded from local files.")

except Exception as e:
    print(f"   ERROR: {e}")
    print("   Tip: For OPeNDAP, ensure netcdf4 is installed: pip install netcdf4")
    exit()

# --- 2. INITIALIZE PARTICLES ---
if USE_RIVER_SOURCES:
    from river_data import get_particle_start_positions
    print("2. Loading river plastic emissions (Meijer 2021)...")
    particles_x, particles_y = get_particle_start_positions(
        NUM_PARTICLES,
        bbox=RIVER_BBOX,
        min_emission=0.1,
    )
    print(f"   Placed {NUM_PARTICLES} particles at river mouths (emission-weighted)")
else:
    particles_x = np.full(NUM_PARTICLES, START_LON)
    particles_y = np.linspace(START_LAT_MIN, START_LAT_MAX, NUM_PARTICLES)
    print(f"2. Dropping {NUM_PARTICLES} particles in a line east of Japan...")

def nudge_to_ocean(px, py, lon_2d, lat_2d, land_mask):
    """Move particles that start on land to nearest ocean cell."""
    ocean = ~land_mask
    for i in range(len(px)):
        x, y = px[i], py[i]
        xi = int(np.clip((np.abs(lon_2d[0, :] - x)).argmin(), 0, lon_2d.shape[1] - 1))
        yi = int(np.clip((np.abs(lat_2d[:, 0] - y)).argmin(), 0, lon_2d.shape[0] - 1))
        if yi >= land_mask.shape[0] or xi >= land_mask.shape[1]:
            continue
        if land_mask[yi, xi] or not (np.isfinite(lon_2d[yi, xi]) and np.isfinite(lat_2d[yi, xi])):
            for r in range(1, max(lon_2d.shape)):
                ylo, yhi = max(0, yi - r), min(lon_2d.shape[0], yi + r + 1)
                xlo, xhi = max(0, xi - r), min(lon_2d.shape[1], xi + r + 1)
                patch = ocean[ylo:yhi, xlo:xhi]
                if np.any(patch):
                    yy, xx = np.where(patch)
                    dist = (lat_2d[ylo:yhi, xlo:xhi][yy, xx] - y)**2 + (lon_2d[ylo:yhi, xlo:xhi][yy, xx] - x)**2
                    j = np.argmin(dist)
                    px[i] = float(lon_2d[ylo + yy[j], xlo + xx[j]])
                    py[i] = float(lat_2d[ylo + yy[j], xlo + xx[j]])
                    break

# Build 2D lon/lat and nudge particles from land to ocean
if np.ndim(lons) == 1:
    lon_2d, lat_2d = np.meshgrid(lons, lats)
else:
    lon_2d, lat_2d = lons, lats
land_mask = np.isnan(u_vis)
nudge_to_ocean(particles_x, particles_y, lon_2d, lat_2d, land_mask)
print("   Nudged land particles to nearest ocean")

# --- 3. PHYSICS ENGINE ---
# When on land: IMMEDIATELY displace to nearest ocean (velocity-based nudge wasn't reliable)
DISPLACE_DEG = 0.25  # move this many degrees toward ocean when on land (~3 grid cells)

def _nearest_ocean_cell(x, y, x_axis, y_axis, land_mask):
    """Return (xo, yo) of nearest ocean cell, or None if none found."""
    xi = int(np.clip((np.abs(x_axis[0, :] - x)).argmin(), 0, x_axis.shape[1] - 1))
    yi = int(np.clip((np.abs(y_axis[:, 0] - y)).argmin(), 0, y_axis.shape[0] - 1))
    ocean = ~land_mask
    ny, nx = land_mask.shape
    for r in range(0, max(nx, ny)):
        ylo, yhi = max(0, yi - r), min(ny, yi + r + 1)
        xlo, xhi = max(0, xi - r), min(nx, xi + r + 1)
        patch = ocean[ylo:yhi, xlo:xhi]
        if np.any(patch):
            yy, xx = np.where(patch)
            dists = (x_axis[ylo:yhi, xlo:xhi][yy, xx] - x)**2 + (y_axis[ylo:yhi, xlo:xhi][yy, xx] - y)**2
            j = np.argmin(dists)
            return float(x_axis[ylo + yy[j], xlo + xx[j]]), float(y_axis[ylo + yy[j], xlo + xx[j]])
    return None

def _direction_away_from_land(x, y, x_axis, y_axis, land_mask):
    """When at coast, return (xo, yo) to move toward - direction away from nearest land."""
    xi = int(np.clip((np.abs(x_axis[0, :] - x)).argmin(), 0, x_axis.shape[1] - 1))
    yi = int(np.clip((np.abs(y_axis[:, 0] - y)).argmin(), 0, y_axis.shape[0] - 1))
    ny, nx = land_mask.shape
    for r in range(1, max(nx, ny)):
        ylo, yhi = max(0, yi - r), min(ny, yi + r + 1)
        xlo, xhi = max(0, xi - r), min(nx, xi + r + 1)
        patch = land_mask[ylo:yhi, xlo:xhi]
        if np.any(patch):
            yy, xx = np.where(patch)
            dists = (x_axis[ylo:yhi, xlo:xhi][yy, xx] - x)**2 + (y_axis[ylo:yhi, xlo:xhi][yy, xx] - y)**2
            j = np.argmin(dists)
            xl, yl = float(x_axis[ylo + yy[j], xlo + xx[j]]), float(y_axis[ylo + yy[j], xlo + xx[j]])
            # Move away from land: target = particle + (particle - land) = 2*particle - land
            xo = 2 * x - xl
            yo = 2 * y - yl
            return xo, yo
    return None

def get_velocity_at_point(x, y, u_grid, v_grid, x_axis, y_axis, land_mask):
    """Get velocity at (x,y). If on land, return zero (displacement handled separately)."""
    xi = int(np.clip((np.abs(x_axis[0, :] - x)).argmin(), 0, x_axis.shape[1] - 1))
    yi = int(np.clip((np.abs(y_axis[:, 0] - y)).argmin(), 0, y_axis.shape[0] - 1))
    on_land = land_mask[yi, xi] if (yi < land_mask.shape[0] and xi < land_mask.shape[1]) else False
    invalid = np.isnan(u_grid[yi, xi]) or np.isnan(v_grid[yi, xi]) or abs(u_grid[yi, xi]) > 10 or abs(v_grid[yi, xi]) > 10
    if on_land or invalid:
        return 0.0, 0.0  # Velocity handled by displacement below
    return float(u_grid[yi, xi]), float(v_grid[yi, xi])

history_x = []
history_y = []

# --- 4. RUN SIMULATION ---
dt = 6 * 3600  # 6-hour steps
meters_per_degree = 111000.0

# For time-varying: 3-hourly data -> 2 velocity steps per 6-hour sim step
steps_per_velocity_update = 2 if time_varying else 1
chunk_idx = 0
if time_varying:
    u_chunk, v_chunk = u_all, v_all  # First chunk already loaded

print(f"3. Simulating {DAYS_TO_SIMULATE} days...")
total_steps = DAYS_TO_SIMULATE * 4

for step in range(total_steps):
    if time_varying:
        velocity_idx = step * steps_per_velocity_update
        # Load next chunk when we exceed current chunk
        if velocity_idx >= (chunk_idx + 1) * CHUNK_STEPS:
            chunk_idx += 1
            base = step_2024 if source == "analysis" else 0
            step_start = base + chunk_idx * CHUNK_STEPS
            step_end = min(step_start + CHUNK_STEPS, base + DAYS_TO_SIMULATE * 8)
            if step_start < step_end:
                u_chunk, v_chunk, _, _ = load_velocity_chunk(
                    source, step_start, step_end, bbox=BBOX, stride=OPENDAP_STRIDE
                )
                print(f"   Loaded days {chunk_idx*CHUNK_DAYS}-{chunk_idx*CHUNK_DAYS + u_chunk.shape[0]//8}")
        idx_in_chunk = velocity_idx - chunk_idx * CHUNK_STEPS
        idx_in_chunk = min(idx_in_chunk, u_chunk.shape[0] - 1)
        u_curr_grid = u_chunk[idx_in_chunk]
        v_curr_grid = v_chunk[idx_in_chunk]
    else:
        u_curr_grid = u_data
        v_curr_grid = v_data

    for i in range(NUM_PARTICLES):
        u_curr, v_curr = get_velocity_at_point(
            particles_x[i], particles_y[i], u_curr_grid, v_curr_grid, lon_2d, lat_2d, land_mask
        )
        dx = u_curr * dt
        dy = v_curr * dt
        particles_x[i] += dx / (meters_per_degree * np.cos(np.radians(particles_y[i])))
        particles_y[i] += dy / meters_per_degree

        # If on land, invalid cell, or adjacent to land (at coast): displace toward ocean
        xi = int(np.clip((np.abs(lon_2d[0, :] - particles_x[i])).argmin(), 0, lon_2d.shape[1] - 1))
        yi = int(np.clip((np.abs(lat_2d[:, 0] - particles_y[i])).argmin(), 0, lat_2d.shape[0] - 1))
        invalid = yi < u_curr_grid.shape[0] and xi < u_curr_grid.shape[1] and (
            np.isnan(u_curr_grid[yi, xi]) or np.isnan(v_curr_grid[yi, xi]) or
            abs(u_curr_grid[yi, xi]) > 10 or abs(v_curr_grid[yi, xi]) > 10
        )
        on_land = yi < land_mask.shape[0] and xi < land_mask.shape[1] and land_mask[yi, xi]
        # Also displace if adjacent to land (particle at coast in ocean cell)
        at_coast = False
        if not on_land and yi < land_mask.shape[0] and xi < land_mask.shape[1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = yi + dy, xi + dx
                    if 0 <= ny < land_mask.shape[0] and 0 <= nx < land_mask.shape[1] and land_mask[ny, nx]:
                        at_coast = True
                        break
        if on_land or invalid or at_coast:
            if at_coast and not on_land:
                target = _direction_away_from_land(particles_x[i], particles_y[i], lon_2d, lat_2d, land_mask)
            else:
                target = _nearest_ocean_cell(particles_x[i], particles_y[i], lon_2d, lat_2d, land_mask)
            if target is not None:
                xo, yo = target
                dist = max(np.sqrt((xo - particles_x[i])**2 + (yo - particles_y[i])**2), 0.01)
                step_frac = min(DISPLACE_DEG / dist, 1.0)
                particles_x[i] += step_frac * (xo - particles_x[i])
                particles_y[i] += step_frac * (yo - particles_y[i])

    if step % 4 == 0:
        history_x.append(particles_x.copy())
        history_y.append(particles_y.copy())
        if (step / 4) % 30 == 0:
            print(f"   ... Day {int(step/4)} complete")

# u_vis, v_vis set during load (first snapshot for map background)
print("4. Generating Animation...")

# --- 5. VISUALIZATION ---
# Pacific only: Japan, gyre, Kuroshio (no cartopy - it keeps resetting to global view)
MAP_EXTENT = [110, 200, 15, 50]  # lon_min, lon_max, lat_min, lat_max

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(*MAP_EXTENT[:2])
ax.set_ylim(*MAP_EXTENT[2:])
ax.set_autoscalex_on(False)
ax.set_autoscaley_on(False)
ax.set_facecolor("#E0F0FF")

# Land from HYCOM mask (only in our data region)
land_mask = np.isnan(u_vis)
ax.contourf(lon_2d, lat_2d, land_mask.astype(float), levels=[0.5, 1.5], colors=["#D2B48C"], zorder=2)
ax.contour(lon_2d, lat_2d, land_mask.astype(float), levels=[0.5], colors="black", linewidths=0.5, zorder=3)

quiv_stride = 20
ax.quiver(lon_2d[::quiv_stride, ::quiv_stride], lat_2d[::quiv_stride, ::quiv_stride],
          u_vis[::quiv_stride, ::quiv_stride], v_vis[::quiv_stride, ::quiv_stride],
          color="lightgray", alpha=0.6, scale=50, width=0.003, zorder=1)
scatter = ax.scatter([], [], c="red", s=25, edgecolor="black", zorder=5)
title = ax.set_title("Day 0: The Launch")
ax.set_xlabel("Longitude (Degrees East)")
ax.set_ylabel("Latitude")

def update(frame):
    scatter.set_offsets(np.c_[history_x[frame], history_y[frame]])
    title.set_text(f"Day {frame}: Tracking the Debris")
    ax.set_xlim(*MAP_EXTENT[:2])
    ax.set_ylim(*MAP_EXTENT[2:])
    return scatter, title

anim = FuncAnimation(fig, update, frames=len(history_x), interval=50, blit=False)

# Fill window edge-to-edge (remove white margins)
plt.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.04)

if SAVE_GIF:
    print("Saving animation to GIF (this may take a minute)...")
    try:
        anim.save(GIF_FILENAME, writer="pillow", fps=15)
        print(f"Saved to {GIF_FILENAME}")
    except Exception as e:
        print(f"GIF save failed: {e}")
        print("Install Pillow: pip install pillow")

print("Done! Check the pop-up window.")
plt.show()
