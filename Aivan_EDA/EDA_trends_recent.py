"""
Trend Analysis: 2019-2025 Averages (Recent Data)

Fast script (~5-10 min): Loads 1 month per year from GOFS 3.1 Analysis (2018-2024),
computes mean current speed in the North Pacific Gyre, and plots trends.

Uses OPeNDAP - no API key, no large download.
"""

import numpy as np
import matplotlib.pyplot as plt
from hycom_data import load_hycom_opendap, get_step_range_for_year, STEPS_PER_YEAR

# North Pacific gyre region
BBOX = (140, 220, 25, 45)

# Gyre core for averaging (where plastic accumulates)
GYRE_BBOX = (180, 220, 30, 40)  # Central Pacific

# Years to analyze
YEARS = [2019, 2020, 2021, 2022, 2023, 2024]

# Month to sample (7 = July = summer)
MONTH = 7

# Steps per month (~3-hourly: 31*8 = 248)
STEPS_PER_MONTH = 248

print("=" * 55)
print("Trend Analysis: Recent Years (2019-2024)")
print("=" * 55)
print("\nLoading 1 month per year from GOFS 3.1 Analysis...")
print("(6 small requests - should complete in ~5-10 min)\n")

results = {}
yearly_speeds = []

for year in YEARS:
    step_start, step_end = get_step_range_for_year("analysis", year)
    if step_end <= step_start:
        print(f"   Skip {year}: no data")
        continue

    # July: ~6 months into year = 6*248 = 1488 steps from year start
    month_offset = (MONTH - 1) * STEPS_PER_MONTH
    month_start = step_start + month_offset
    month_end = min(month_start + STEPS_PER_MONTH, step_end)

    if month_end <= month_start:
        print(f"   Skip {year}: month {MONTH} out of range")
        continue

    print(f"   Loading {year} July...")
    try:
        ds = load_hycom_opendap(
            source="analysis",
            time_slice=slice(month_start, month_end),
            bbox=BBOX,
        )
        u = ds["u"].values
        v = ds["v"].values
        lons = ds["lon"].values
        lats = ds["lat"].values

        speed = np.sqrt(u**2 + v**2)
        mean_speed = float(np.nanmean(speed))
        results[year] = mean_speed
        yearly_speeds.append(mean_speed)

        # Gyre region mean if we have the subset
        if lons.ndim == 1:
            lon_2d, lat_2d = np.meshgrid(lons, lats)
        else:
            lon_2d, lat_2d = lons, lats
        gyre_mask = (
            (lon_2d >= GYRE_BBOX[0]) & (lon_2d <= GYRE_BBOX[1])
            & (lat_2d >= GYRE_BBOX[2]) & (lat_2d <= GYRE_BBOX[3])
        )
        gyre_speed = float(np.nanmean(speed[:, gyre_mask]))
        results[f"{year}_gyre"] = gyre_speed

        try:
            ds.close()
        except Exception:
            pass
    except Exception as e:
        print(f"   Error {year}: {e}")

if not results:
    print("No data loaded. Check connection.")
    exit()

# --- 1. Year-over-year trend ---
fig, axes = plt.subplots(2, 1, figsize=(10, 8))
fig.suptitle("North Pacific Current Trends (2019-2024)", fontsize=14)

ax1 = axes[0]
years = [y for y in YEARS if y in results]
speeds = [results[y] for y in years]
ax1.bar(years, speeds, color="#2E86AB", edgecolor="black")
ax1.set(xlabel="Year", ylabel="Mean Speed (m/s)", title="July Mean Current Speed")
ax1.set_ylim(0, None)

# --- 2. Trend line ---
ax2 = axes[1]
gyre_speeds = [results.get(f"{y}_gyre", results[y]) for y in years]
ax2.plot(years, gyre_speeds, "o-", color="#C73E1D", linewidth=2, markersize=10)
ax2.set(xlabel="Year", ylabel="Gyre Core Speed (m/s)", title="Gyre Region Trend")
ax2.grid(True, alpha=0.3)

# Simple trend
if len(years) >= 2:
    z = np.polyfit(years, gyre_speeds, 1)
    p = np.poly1d(z)
    ax2.plot(years, p(years), "--", color="gray", alpha=0.7, label=f"Trend: {z[0]:.4f}/yr")

plt.tight_layout()
print("\nDone! Close the window to exit.")
plt.show()
