import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

print("1. Loading data...")
BBOX = (130, 240, 18, 50)  # North Pacific

# Try local files first, then OPeNDAP
if Path("2018_uvel.nc4").exists() and Path("2018_vvel.nc4").exists():
    ds_u = xr.open_dataset("2018_uvel.nc4", engine="h5netcdf")
    ds_v = xr.open_dataset("2018_vvel.nc4", engine="h5netcdf")
    u = ds_u["u"].squeeze()
    v = ds_v["v"].squeeze()
    time_dim = "MT" if "MT" in u.dims else "time"
    u_snap = u.isel({time_dim: 0}).values
    v_snap = v.isel({time_dim: 0}).values
    lons = ds_u.Longitude.values
    lats = ds_u.Latitude.values
    if lons.ndim == 1:
        lons, lats = np.meshgrid(lons, lats)
    print("   Loaded from local files.")
else:
    from hycom_data import load_hycom_opendap
    ds = load_hycom_opendap(source="reanalysis", time_slice=slice(0, 100), bbox=BBOX)
    u_snap = ds["u"].isel(time=0).values
    v_snap = ds["v"].isel(time=0).values
    lons = ds["lon"].values
    lats = ds["lat"].values
    if lons.ndim == 1:
        lons, lats = np.meshgrid(lons, lats)
    print("   Loaded via OPeNDAP (no local files).")

print("2. Calculating current speeds...")
speed = np.sqrt(u_snap**2 + v_snap**2)

print("3. Generating plot...")
fig, ax = plt.subplots(1, 2, figsize=(18, 8))

# --- PLOT 1: Speed Heatmap ---
ax[0].pcolormesh(lons, lats, speed, cmap="viridis", vmin=0, vmax=1.5)
ax[0].set_title("Current Speed (Blue = Plastic Traps)")
ax[0].set_xlabel("Longitude")
ax[0].set_ylabel("Latitude")
fig.colorbar(ax[0].collections[0], ax=ax[0], label="Current Speed (m/s)")

# --- PLOT 2: Vector Field ---
step = 10
X = lons[::step, ::step]
Y = lats[::step, ::step]
U_sub = u_snap[::step, ::step]
V_sub = v_snap[::step, ::step]
ax[1].quiver(X, Y, U_sub, V_sub, scale=20, width=0.002)
ax[1].set_title("Surface Currents (Look for the Gyre)")
ax[1].set_xlabel("Longitude")
ax[1].set_ylabel("Latitude")

print("Done! Check the window.")
plt.show()
