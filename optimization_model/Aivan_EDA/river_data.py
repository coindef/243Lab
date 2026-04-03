"""
River Plastic Emissions - Meijer et al. 2021

Loads river outfall locations and emissions from:
  Meijer et al. "More than 1000 rivers account for 80% of global riverine
  plastic emissions into the ocean" (2021)
  https://figshare.com/articles/dataset/Supplementary_data_for_More_than_1000_rivers_account_for_80_of_global_riverine_plsatic_emissions_into_the_ocean_/14515590

31,819 river outfalls globally. Emissions in metric tons/year.
"""

import os
import zipfile
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Figshare file ID for Meijer2021_midpoint_emissions.zip
FIGSHARE_FILE_ID = "27807774"
FIGSHARE_URL = f"https://ndownloader.figshare.com/files/{FIGSHARE_FILE_ID}"
DATA_DIR = Path(__file__).parent / "data"
ZIP_PATH = DATA_DIR / "Meijer2021_midpoint_emissions.zip"

# North Pacific: rivers from Asia that feed into the gyre
# Japan, China, Korea, Taiwan, Philippines, Vietnam, etc.
NORTH_PACIFIC_BBOX = (100, 180, 0, 55)  # lon_min, lon_max, lat_min, lat_max


def _download_dataset() -> Path:
    """Download the Meijer dataset from figshare."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists():
        return ZIP_PATH
    print(f"   Downloading Meijer river emissions (~700 KB)...")
    try:
        req = urllib.request.Request(FIGSHARE_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        if len(data) < 1000 or not data[:4] == b"PK\x03\x04":  # ZIP magic
            raise ValueError("Downloaded file is not a valid zip")
        with open(ZIP_PATH, "wb") as f:
            f.write(data)
        print(f"   Saved to {ZIP_PATH}")
    except Exception as e:
        raise RuntimeError(
            f"Download failed: {e}\n"
            f"Download manually from:\n"
            f"  https://figshare.com/ndownloader/files/{FIGSHARE_FILE_ID}\n"
            f"  Save as: {ZIP_PATH}"
        ) from e
    return ZIP_PATH


def load_river_emissions(
    bbox: Optional[Tuple[float, float, float, float]] = None,
    min_emission: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load river outfall locations and emissions for the North Pacific.

    Parameters
    ----------
    bbox : tuple, optional
        (lon_min, lon_max, lat_min, lat_max). Default: North Pacific.
    min_emission : float
        Minimum emission (tons/year) to include.

    Returns
    -------
    lons, lats, emissions : arrays
        Longitude, latitude, and emissions (tons/year) for each river.
    """
    bbox = bbox or NORTH_PACIFIC_BBOX
    lon_min, lon_max, lat_min, lat_max = bbox

    # Ensure dataset exists
    if not ZIP_PATH.exists():
        _download_dataset()

    try:
        import geopandas as gpd
    except ImportError:
        raise ImportError(
            "geopandas required for shapefile. Install: pip install geopandas"
        ) from None

    # Find shapefile in zip
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        shp_files = [n for n in zf.namelist() if n.endswith(".shp")]
        if not shp_files:
            raise ValueError(f"No .shp found in {ZIP_PATH}")
        # Extract to temp dir and read
        extract_dir = DATA_DIR / "meijer_extract"
        extract_dir.mkdir(parents=True, exist_ok=True)
        zf.extractall(extract_dir)
        shp_path = extract_dir / shp_files[0]

    gdf = gpd.read_file(shp_path)

    # Get coordinates from geometry
    gdf = gdf.to_crs("EPSG:4326")  # WGS84
    gdf["lon"] = gdf.geometry.x
    gdf["lat"] = gdf.geometry.y

    # Find emission column (Meijer uses various names)
    em_col = None
    for name in ["Emissions_t", "emissions", "Emissions", "tonnes", "tons", "Mton", "value"]:
        if name in gdf.columns:
            em_col = name
            break
    if em_col is None:
        numeric = gdf.select_dtypes(include=[np.number]).columns
        em_col = numeric[0] if len(numeric) > 0 else gdf.columns[0]

    emissions = gdf[em_col].fillna(0).values

    # Filter by bbox and min emission
    mask = (
        (gdf["lon"] >= lon_min)
        & (gdf["lon"] <= lon_max)
        & (gdf["lat"] >= lat_min)
        & (gdf["lat"] <= lat_max)
        & (emissions >= min_emission)
    )
    gdf = gdf[mask]
    emissions = emissions[mask]

    lons = gdf["lon"].values
    lats = gdf["lat"].values

    return lons, lats, emissions


def get_particle_start_positions(
    n_particles: int,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    min_emission: float = 0.1,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get (lons, lats) for n_particles, weighted by river emissions.

    Particles are placed at river mouths, with more particles at high-emission rivers.
    """
    lons, lats, emissions = load_river_emissions(bbox=bbox, min_emission=min_emission)

    if len(lons) == 0:
        raise ValueError("No rivers in bbox. Try wider bbox or lower min_emission.")

    # Weight by emissions (normalize to probabilities)
    weights = emissions / emissions.sum()
    rng = np.random.default_rng(random_seed)
    indices = rng.choice(len(lons), size=n_particles, replace=True, p=weights)

    return lons[indices], lats[indices]
