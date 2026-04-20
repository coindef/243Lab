"""
================================================================================
Backend Interface for Plastics Hunter Dashboard
================================================================================

This module is the bridge between the dashboard (`app.py`) and the optimization
model produced by the team's research notebook.

--------------------------------------------------------------------------------
HOW TO UPDATE THE BACKEND MODEL (for teammate A)
--------------------------------------------------------------------------------

1. Re-run your research notebook (`main_optimization_model_*.ipynb`) and export
   a fresh bundle file named EXACTLY:

       optimization_dashboard_bundle.pkl

   The bundle must be a dict pickled with `cloudpickle` containing these keys:

       bundle["meta"]        -> dict with BBOX, PORT_LON, PORT_LAT,
                                 n_prod_snapshots, ...
       bundle["query"]       -> callable(departure_day, port, trip_days,
                                 n_vessels) -> (routes, plastic, fuel, forecast)
       bundle["find_best"]   -> callable(port, trip_days, candidate_days,
                                 n_vessels) -> list of dicts with keys:
                                 departure_day, plastic_collected,
                                 fuel_consumed, total_mass, normalized_plastic

2. Upload the new pickle file to Google Drive.

   - IMPORTANT: To avoid changing the file ID, REPLACE the existing file using
     "Manage versions" (right-click the file in Drive -> Manage versions ->
     Upload new version). This keeps the same shareable ID.

   - If you must upload a brand new file, update BUNDLE_GDRIVE_ID below with the
     new file ID and commit to GitHub. The Streamlit Cloud deployment will
     auto-redeploy.

3. (optional) If you change the bundle schema (e.g. rename "query" to
   "compute_routes"), update the wrapper functions in this file accordingly.

--------------------------------------------------------------------------------
"""
import os
import numpy as np

# ----------------------------------------------------------------------------
# Configuration — Google Drive file ID for the optimization bundle
# ----------------------------------------------------------------------------
# Paste the file ID from the shareable Google Drive link here.
# Example link:  https://drive.google.com/file/d/1AbCdEfGhIjK.../view?usp=sharing
#                                             ^^^^^^^^^^^^^^^^^^^  <- this part
BUNDLE_GDRIVE_ID = os.environ.get("BUNDLE_GDRIVE_ID", "1_xQOktokI45LVZkCjpUz3rlXIW_ha8Mw")

# Local cache path (relative to this file). The file is downloaded on first run.
BUNDLE_PATH = os.path.join(os.path.dirname(__file__), "data",
                           "optimization_dashboard_bundle.pkl")

# Expected file size in bytes (used to validate download, optional).
# Set to None to skip size validation.
BUNDLE_EXPECTED_SIZE = None  # e.g. 191_438_270

# ----------------------------------------------------------------------------
# SE Asia port presets (editable by UI designer)
# ----------------------------------------------------------------------------
PORTS = {
    "Ho Chi Minh City": (106.7, 10.8),
    "Singapore": (103.8, 1.3),
    "Jakarta": (106.8, -6.1),
    "Bangkok": (100.5, 13.7),
    "Manila": (120.9, 14.6),
    "Hong Kong": (114.2, 22.3),
}

_bundle = None


def _download_bundle_if_needed():
    """Download the bundle from Google Drive on first run."""
    if os.path.exists(BUNDLE_PATH):
        return
    os.makedirs(os.path.dirname(BUNDLE_PATH), exist_ok=True)

    if BUNDLE_GDRIVE_ID == "REPLACE_WITH_GOOGLE_DRIVE_FILE_ID":
        raise RuntimeError(
            "BUNDLE_GDRIVE_ID is not set. Upload the bundle to Google Drive "
            "and paste the file ID into backend.py, OR set the "
            "BUNDLE_GDRIVE_ID environment variable / Streamlit secret."
        )

    try:
        import gdown
    except ImportError as e:
        raise RuntimeError(
            "The 'gdown' package is required to download the bundle. "
            "Add it to requirements.txt."
        ) from e

    print(f"[backend] Downloading bundle from Google Drive (id={BUNDLE_GDRIVE_ID}) ...")
    gdown.download(id=BUNDLE_GDRIVE_ID, output=BUNDLE_PATH, quiet=False)

    if not os.path.exists(BUNDLE_PATH):
        raise RuntimeError(
            f"Failed to download bundle from Google Drive (id={BUNDLE_GDRIVE_ID}). "
            "Check that the file is shared as 'Anyone with the link'."
        )

    if BUNDLE_EXPECTED_SIZE is not None:
        actual = os.path.getsize(BUNDLE_PATH)
        if actual < BUNDLE_EXPECTED_SIZE * 0.9:
            raise RuntimeError(
                f"Downloaded bundle is too small ({actual} bytes, expected "
                f"~{BUNDLE_EXPECTED_SIZE}). The sharing link may be wrong."
            )


def load_bundle():
    """Load the optimization bundle (cached in memory after first call)."""
    global _bundle
    if _bundle is not None:
        return _bundle

    _download_bundle_if_needed()

    import cloudpickle
    with open(BUNDLE_PATH, "rb") as f:
        _bundle = cloudpickle.load(f)
    return _bundle


def get_meta():
    """Return bundle metadata (BBOX, port defaults, snapshot count, etc.)."""
    return load_bundle()["meta"]


def query_route(departure_day, port, trip_days, n_vessels=2):
    """
    Compute optimal routes for a given departure day.

    Parameters
    ----------
    departure_day : int
        Day-of-year index (0-based into the simulation snapshots).
    port : tuple of (float, float)
        (lon, lat) of the departure port.
    trip_days : int
        Forecast horizon (days) used for density prediction.
    n_vessels : int
        Number of vessels to route.

    Returns
    -------
    routes : list of list of (lon, lat) tuples, one list per vessel
    plastic_collected : float     (mass x sweep-area score)
    fuel_consumed : float         (degrees)
    forecast_density : np.ndarray (shape: [lat, lon, time])
    """
    bundle = load_bundle()
    return bundle["query"](
        departure_day=departure_day,
        port=port,
        trip_days=trip_days,
        n_vessels=n_vessels,
    )


def find_best_departure(port, trip_days, day_start, day_end, n_vessels=2):
    """
    Search for the best departure day in an inclusive range.

    Returns a list of dicts sorted by `normalized_plastic` descending.
    Each dict has:
        departure_day, plastic_collected, fuel_consumed,
        total_mass, normalized_plastic
    """
    bundle = load_bundle()
    meta = bundle["meta"]
    max_day = meta["n_prod_snapshots"]
    candidate_days = list(range(
        max(0, day_start),
        min(max_day, day_end + 1),
    ))
    if not candidate_days:
        return []
    return bundle["find_best"](
        port=port,
        trip_days=trip_days,
        candidate_days=candidate_days,
        n_vessels=n_vessels,
    )


def fuel_deg_to_km(fuel_deg):
    """Convert a distance expressed in degrees to approximate kilometres."""
    return fuel_deg * 111.0
