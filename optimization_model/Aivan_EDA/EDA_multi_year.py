"""
Multi-Year EDA: Seasonality & Trends in North Pacific Currents

Uses HYCOM GOFS 3.1 Reanalysis (1994-2015) via OPeNDAP—no download required.
Explores:
  - Seasonal patterns (winter vs summer gyre strength)
  - Year-over-year trends
  - Monthly climatology
"""

import numpy as np
import matplotlib.pyplot as plt
from hycom_data import load_hycom_opendap

# North Pacific region (tighter = less data, avoids OPeNDAP timeout)
BBOX = (140, 220, 25, 45)
# Spatial stride: every 2nd point = 4x less data
STRIDE = 2

# Years to analyze
YEARS = [1995, 2000, 2005, 2010, 2015]

# Months for seasonal comparison (Jan, Apr, Jul, Oct)
SEASONAL_MONTHS = [1, 4, 7, 10]

# Steps per year, per month (3-hourly = 8/day)
STEPS_PER_YEAR = 2920
STEPS_PER_MONTH = 248

print("=" * 60)
print("Multi-Year EDA: Seasonality & Trends")
print("=" * 60)
print("\n1. Connecting to HYCOM Reanalysis (1994-2015)...")
print("   Loading one month at a time to avoid server timeout.\n")

results = {}  # (year, month) -> mean_speed_over_region
yearly_means = {}
snapshot_data = None

for year in YEARS:
    year_speeds = []
    for month in SEASONAL_MONTHS:
        step_start = (year - 1994) * STEPS_PER_YEAR + (month - 1) * STEPS_PER_MONTH
        step_end = min(step_start + STEPS_PER_MONTH, 63341)
        if step_end <= step_start:
            continue

        print(f"   Loading {year} {['Jan','Apr','Jul','Oct'][SEASONAL_MONTHS.index(month)]}...")
        try:
            ds = load_hycom_opendap(
                source="reanalysis",
                time_slice=slice(step_start, step_end),
                bbox=BBOX,
                decode_times=False,
            )
            if STRIDE > 1:
                ds = ds.isel(lon=slice(None, None, STRIDE), lat=slice(None, None, STRIDE))
            u = ds["u"].values
            v = ds["v"].values
            speed = np.sqrt(u**2 + v**2)
            mean_spd = float(np.nanmean(speed))
            results[(year, month)] = mean_spd
            year_speeds.append(mean_spd)
            if month == 7:
                snapshot_data = {
                    "u": np.nanmean(u, axis=0),
                    "v": np.nanmean(v, axis=0),
                    "lon": ds["lon"].values,
                    "lat": ds["lat"].values,
                    "year": year,
                }
            try:
                ds.close()
            except Exception:
                pass
        except Exception as e:
            print(f"   Skip {year}-{month}: {e}")

    if year_speeds:
        yearly_means[year] = float(np.mean(year_speeds))

print("\n2. Computing statistics...")

# --- PLOTTING ---
print("\n3. Generating plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Multi-Year EDA: North Pacific Currents (HYCOM Reanalysis 1994-2015)", fontsize=12)

# Plot 1: Seasonal comparison (use first available year)
ax1 = axes[0, 0]
month_names = ["Jan", "Apr", "Jul", "Oct"]
ref_year = YEARS[-1] if YEARS else 2010
seasonal_speeds = [results.get((ref_year, m), np.nan) for m in SEASONAL_MONTHS]
colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]
bars = ax1.bar(month_names, seasonal_speeds, color=colors)
ax1.set_ylabel("Mean Current Speed (m/s)")
ax1.set_title(f"Seasonality: {ref_year} (Winter → Summer)")
ax1.set_ylim(0, None)

# Plot 2: Year-over-year trend
ax2 = axes[0, 1]
yr_list = sorted(yearly_means.keys())
spd_list = [yearly_means[y] for y in yr_list]
ax2.plot(yr_list, spd_list, "o-", color="#2E86AB", linewidth=2, markersize=8)
ax2.set_xlabel("Year")
ax2.set_ylabel("Mean Annual Speed (m/s)")
ax2.set_title("Year-over-Year Trend")
ax2.grid(True, alpha=0.3)

# Plot 3: Seasonal heatmap across years
ax3 = axes[1, 0]
matrix = np.full((len(YEARS), len(SEASONAL_MONTHS)), np.nan)
for i, y in enumerate(YEARS):
    for j, m in enumerate(SEASONAL_MONTHS):
        matrix[i, j] = results.get((y, m), np.nan)
im = ax3.imshow(matrix, aspect="auto", cmap="viridis")
ax3.set_xticks(range(4))
ax3.set_xticklabels(month_names)
ax3.set_yticks(range(len(YEARS)))
ax3.set_yticklabels([str(y) for y in YEARS])
ax3.set_xlabel("Month")
ax3.set_ylabel("Year")
ax3.set_title("Mean Speed by Year & Season")
plt.colorbar(im, ax=ax3, label="Speed (m/s)")

# Plot 4: Single snapshot - current speed map (like original EDA)
ax4 = axes[1, 1]
if snapshot_data:
    u_snap = snapshot_data["u"]
    v_snap = snapshot_data["v"]
    lons = snapshot_data["lon"]
    lats = snapshot_data["lat"]
    snap_year = snapshot_data.get("year", "?")
else:
    u_snap = np.zeros((10, 10))
    v_snap = np.zeros((10, 10))
    lons = np.linspace(130, 240, 10)
    lats = np.linspace(18, 50, 10)
    snap_year = "?"
speed_snap = np.sqrt(u_snap**2 + v_snap**2)
if np.ndim(lons) == 1:
    lon_2d, lat_2d = np.meshgrid(lons, lats)
else:
    lon_2d, lat_2d = lons, lats
pc = ax4.pcolormesh(lon_2d, lat_2d, speed_snap, cmap="viridis", vmin=0, vmax=1.5)
ax4.set_xlabel("Longitude")
ax4.set_ylabel("Latitude")
ax4.set_title(f"Mean July {snap_year} Speed (Gyre Region)")
plt.colorbar(pc, ax=ax4, label="Speed (m/s)")

plt.tight_layout()
print("\nDone! Close the window to exit.")
plt.show()
