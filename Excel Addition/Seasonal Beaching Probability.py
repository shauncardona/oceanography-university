"""
Seasonal Beaching Probability Grids from Oil-Spill Trajectory CSVs
====================================================================

DATA LAYOUT (matches what you showed):

    <INPUT_ROOT>/
        autumn_2025-11-21/
            run1_0600_36.394N_14.537E_heavy_fuel_o..._GE....csv
            run2_0500_35.951N_14.715E_crude_oil_GE....csv
            ... (15 runs)
        spring_2025-03-01/
            run1_...csv
            ...
        ...

    Each folder name is "<season>_<date>" - the season is taken directly
    from the part before the first underscore, so no date parsing is
    needed to know the season.

    Each CSV has columns: trajectory, time, longitude, latitude, status
        - trajectory: particle ID (resets to 0 within each run file)
        - status: 0 = afloat, 1 = beached/touched land (stays 1 once set)

WHAT THIS SCRIPT COMPUTES

For each season, across every day-folder and every one of the 15 runs
per day:

    - denominator = total number of particles simulated that season
                    (sum of unique `trajectory` IDs per run file)
    - for each particle, if it ever reaches status == 1, its EARLIEST
      such row (first beaching position) is taken as its beaching cell
    - numerator per 100 m grid cell = count of particles whose first
      beaching position falls in that cell
    - cell value = numerator / denominator  -> beaching probability
      (0-1) for that cell, for that season

Cells with zero recorded beaching are written as NoData (transparent), not 0 -
see MASK_ZERO_AS_NODATA below. Since beaching can only physically occur right
at the coast, this means open sea and inland cells drop out entirely and only
cells where a particle actually beached remain - so when you load the GeoTIFF
into ArcGIS/QGIS you get a coastline-hugging layer (like a CVI map) rather
than a solid block covering the whole study area. Classify the remaining
values into 5 quantile/manual bins and colour them blue -> red to match a
"very low -> very high vulnerability" style legend.

In addition to the 4 per-season GeoTIFFs, an "all seasons combined" GeoTIFF
is also produced, pooling every particle from every season into one
probability surface.

Requires: pandas, numpy, scipy, rasterio, pyproj, tqdm
    pip install pandas numpy scipy rasterio pyproj tqdm --break-system-packages
"""

import os
import glob

import numpy as np
import pandas as pd
from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter
import rasterio
from rasterio.transform import from_origin
from pyproj import Transformer
from tqdm import tqdm

# ------------------------- CONFIG -------------------------
INPUT_ROOT = r"E:\University\Applied Oceanography\Dissertation\oceanography-university\Excel Addition\data"        # root folder containing the season_date subfolders
OUTPUT_DIR = r"E:\University\Applied Oceanography\Dissertation\Results\GEOTIFFS"
RESOLUTION = 100                        # grid resolution in metres

LON_COL = "longitude"
LAT_COL = "latitude"
STATUS_COL = "status"
TRAJ_COL = "trajectory"
TIME_COL = "time"

SOURCE_CRS = "EPSG:4326"  # CRS of lat/lon in the CSVs
TARGET_CRS = "EPSG:32633"  # UTM zone 33N - correct metric CRS for Malta

FILE_PATTERN = "*.csv"  # glob pattern for run files inside each folder

# Fix the same bounding box across all 4 seasons so they line up cell-for-cell
# (handy for comparing/overlaying in QGIS later). Covers Malta + a buffer for
# particles that drift out to sea. In EPSG:32633 (UTM 33N) metres.
# Adjust if your runs drift further than this - check a season's console
# output; if beaching events sit right at the edge, widen the box.
MANUAL_BOUNDS = {
    "xmin": 410000, "xmax": 480000,
    "ymin": 3945000, "ymax": 4010000,
}  # None to auto-fit each season's grid to its own beaching-point extent instead

# If True, cells with zero beaching probability become NoData (transparent)
# instead of 0, so only the coastal cells that actually recorded a beaching
# event are drawn - this is what gives you the "dots/line along the coast"
# look rather than a solid rectangle covering land + sea.
MASK_ZERO_AS_NODATA = True

# Gaussian smoothing bandwidth (in metres) for the density/heatmap grids.
# Larger = smoother/broader heatmap, smaller = tighter to the raw points.
KDE_BANDWIDTH_METERS = 500
# ------------------------------------------------------------

SEASONS = ["winter", "spring", "summer", "autumn"]


def season_from_folder(folder_name: str):
    """Season is the token before the first underscore, e.g.
    'autumn_2025-11-21' -> 'autumn'. Returns None if it doesn't match."""
    season = folder_name.split("_")[0].strip().lower()
    return season if season in SEASONS else None


def first_beaching_points(csv_path):
    """
    Read one run CSV and return:
        - n_particles: number of unique trajectories in this run
        - beach_lon, beach_lat: arrays of the FIRST beaching (status==1)
          position for each trajectory that ever beaches
    """
    df = pd.read_csv(csv_path)
    required = [TRAJ_COL, TIME_COL, LON_COL, LAT_COL, STATUS_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"    Skipping {os.path.basename(csv_path)}: missing columns {missing}")
        return 0, np.array([]), np.array([])

    n_particles = df[TRAJ_COL].nunique()

    beached = df[df[STATUS_COL] == 1]
    if beached.empty:
        return n_particles, np.array([]), np.array([])

    # Sort so the first row per trajectory after filtering is the earliest beaching event
    beached = beached.sort_values([TRAJ_COL, TIME_COL])
    first_beach = beached.drop_duplicates(subset=TRAJ_COL, keep="first")

    return n_particles, first_beach[LON_COL].values, first_beach[LAT_COL].values


def collect_season_data(input_root):
    """
    Walk input_root/<season>_<date>/*.csv and return, per season:
        {"n_particles": int, "lon": [...], "lat": [...]}

    Shows a single overall progress bar (files processed / total files,
    with a live ETA) rather than per-folder print spam, since a full run
    can involve hundreds of day-folders x 15 run files each.
    """
    season_data = {s: {"n_particles": 0, "lon": [], "lat": []} for s in SEASONS}

    day_folders = sorted(
        d for d in os.listdir(input_root)
        if os.path.isdir(os.path.join(input_root, d))
    )
    if not day_folders:
        raise FileNotFoundError(f"No subfolders found in {input_root}")

    # First pass (fast - just listing files, not reading them) so we know
    # the total amount of work up front and can show a real percentage/ETA.
    work_items = []
    skipped_folders = []
    for folder in day_folders:
        season = season_from_folder(folder)
        if season is None:
            skipped_folders.append(folder)
            continue
        folder_path = os.path.join(input_root, folder)
        csv_files = sorted(glob.glob(os.path.join(folder_path, FILE_PATTERN)))
        for csv_path in csv_files:
            work_items.append((season, folder, csv_path))

    if skipped_folders:
        print(f"Skipping {len(skipped_folders)} folder(s) with no recognised "
              f"season prefix (e.g. '{skipped_folders[0]}').")

    print(f"Found {len(work_items)} CSV file(s) across "
          f"{len(day_folders) - len(skipped_folders)} day-folder(s). Starting...\n")

    beaching_events_so_far = 0
    progress = tqdm(work_items, desc="Processing runs", unit="file")
    for season, folder, csv_path in progress:
        n_particles, lon, lat = first_beaching_points(csv_path)
        season_data[season]["n_particles"] += n_particles
        if len(lon):
            season_data[season]["lon"].append(lon)
            season_data[season]["lat"].append(lat)
            beaching_events_so_far += len(lon)

        # Live status: current folder + running totals, shown next to the bar
        progress.set_postfix({
            "folder": folder,
            "particles": sum(d["n_particles"] for d in season_data.values()),
            "beached": beaching_events_so_far,
        })

    return season_data


def _bin_beaching_counts(lon, lat, resolution=RESOLUTION):
    """
    Reproject beaching points to TARGET_CRS and bin the COUNT of beaching
    events per `resolution`-metre cell. Shared by both the probability
    grid and the density (smoothed) grid so the binning logic - and the
    grid extent - stays identical between them.
    Returns (count_grid, transform, xmin, ymax).
    """
    transformer = Transformer.from_crs(SOURCE_CRS, TARGET_CRS, always_xy=True)
    x, y = transformer.transform(lon, lat)

    if MANUAL_BOUNDS:
        xmin, xmax = MANUAL_BOUNDS["xmin"], MANUAL_BOUNDS["xmax"]
        ymin, ymax = MANUAL_BOUNDS["ymin"], MANUAL_BOUNDS["ymax"]
    else:
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        # pad by one cell so edge points aren't right on the boundary
        xmin -= resolution
        xmax += resolution
        ymin -= resolution
        ymax += resolution

    xmin = np.floor(xmin / resolution) * resolution
    xmax = np.ceil(xmax / resolution) * resolution
    ymin = np.floor(ymin / resolution) * resolution
    ymax = np.ceil(ymax / resolution) * resolution

    n_cols = max(int((xmax - xmin) / resolution), 1)
    n_rows = max(int((ymax - ymin) / resolution), 1)

    x_edges = np.linspace(xmin, xmax, n_cols + 1)
    y_edges = np.linspace(ymin, ymax, n_rows + 1)

    count, _, _, _ = binned_statistic_2d(
        x, y, None, statistic="count", bins=[x_edges, y_edges]
    )
    # shape (n_cols, n_rows) with x as first axis -> transpose to (rows, cols)
    # and flip vertically so row 0 = north (standard raster row order)
    count = count.T[::-1, :]

    transform = from_origin(xmin, ymax, resolution, resolution)
    return count, transform


def make_probability_grid(lon, lat, n_particles, resolution=RESOLUTION):
    """
    Bin beaching events into a `resolution`-metre grid and divide by
    n_particles to get a raw (unsmoothed) probability grid, with zero
    cells optionally masked to NoData. Returns (grid_array, transform).
    """
    count, transform = _bin_beaching_counts(lon, lat, resolution)

    probability = (count / n_particles).astype("float32") if n_particles > 0 else count.astype("float32")

    if MASK_ZERO_AS_NODATA:
        probability[probability == 0] = np.nan

    return probability, transform


def make_density_grid(lon, lat, n_particles, resolution=RESOLUTION):
    """
    Same binning as make_probability_grid, but the resulting probability
    surface is run through a Gaussian smoothing kernel (bandwidth =
    KDE_BANDWIDTH_METERS) to produce a continuous KDE-style heatmap
    instead of isolated raw cells. Never masked to NoData - the
    smoothing itself fades to ~0 away from beaching clusters.
    Returns (grid_array, transform).
    """
    count, transform = _bin_beaching_counts(lon, lat, resolution)

    probability = (count / n_particles).astype("float32") if n_particles > 0 else count.astype("float32")
    probability = np.nan_to_num(probability, nan=0.0)

    sigma_cells = max(KDE_BANDWIDTH_METERS / resolution, 0.01)
    density = gaussian_filter(probability, sigma=sigma_cells, mode="constant", cval=0.0)

    return density.astype("float32"), transform


def save_geotiff(grid, transform, out_path, nodata=None):
    with rasterio.open(
            out_path, "w",
            driver="GTiff",
            height=grid.shape[0],
            width=grid.shape[1],
            count=1,
            dtype=grid.dtype,
            crs=TARGET_CRS,
            transform=transform,
            nodata=nodata,
            compress="lzw",
    ) as dst:
        dst.write(grid, 1)
    print(f"Saved: {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Reading CSVs from: {INPUT_ROOT}")
    season_data = collect_season_data(INPUT_ROOT)

    print()
    for season in tqdm(SEASONS, desc="Building season grids", unit="season"):
        data = season_data[season]
        n_particles = data["n_particles"]

        if n_particles == 0:
            tqdm.write(f"No particles found for {season}, skipping.")
            continue

        if not data["lon"]:
            tqdm.write(f"{season}: {n_particles} particles simulated, none beached. "
                       f"Skipping grid (all-zero probability everywhere).")
            continue

        lon = np.concatenate(data["lon"])
        lat = np.concatenate(data["lat"])

        tqdm.write(f"{season}: {n_particles} particles simulated, "
                   f"{len(lon)} beaching events -> gridding...")

        grid, transform = make_probability_grid(lon, lat, n_particles)
        out_path = os.path.join(OUTPUT_DIR, f"{season}_beaching_probability_100m.tif")
        save_geotiff(grid, transform, out_path, nodata=(np.nan if MASK_ZERO_AS_NODATA else None))

        density, density_transform = make_density_grid(lon, lat, n_particles)
        density_path = os.path.join(OUTPUT_DIR, f"{season}_beaching_density_100m.tif")
        save_geotiff(density, density_transform, density_path, nodata=None)

    # --- Combined "all seasons" grid: pool every particle from every season ---
    print()
    all_lon_parts = [l for s in SEASONS for l in season_data[s]["lon"]]
    all_particles = sum(season_data[s]["n_particles"] for s in SEASONS)

    if all_particles == 0 or not all_lon_parts:
        print("No data available to build the combined all-seasons grid.")
    else:
        all_lat_parts = [l for s in SEASONS for l in season_data[s]["lat"]]
        combined_lon = np.concatenate(all_lon_parts)
        combined_lat = np.concatenate(all_lat_parts)

        print(f"Combined (all seasons): {all_particles} particles simulated, "
              f"{len(combined_lon)} beaching events -> gridding...")

        grid, transform = make_probability_grid(combined_lon, combined_lat, all_particles)
        out_path = os.path.join(OUTPUT_DIR, "all_seasons_combined_beaching_probability_100m.tif")
        save_geotiff(grid, transform, out_path, nodata=(np.nan if MASK_ZERO_AS_NODATA else None))

        combined_density, combined_density_transform = make_density_grid(
            combined_lon, combined_lat, all_particles
        )
        combined_density_path = os.path.join(
            OUTPUT_DIR, "all_seasons_combined_beaching_density_100m.tif"
        )
        save_geotiff(combined_density, combined_density_transform, combined_density_path, nodata=None)

    print("\nDone.")


if __name__ == "__main__":
    main()
