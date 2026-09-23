# pip install pandas numpy scipy rasterio pyproj tqdm --break-system-packages

import glob
import os

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic_2d
from tqdm import tqdm

# Directory Configuration
INPUT_ROOT = r"E:\University\Applied Oceanography\Dissertation\oceanography-university\Excel Addition\data"
OUTPUT_DIR = r"E:\University\Applied Oceanography\Dissertation\Results\GEOTIFFS"

RESOLUTION = 100  # Grid cell size in meters

# Data Schema Mapping
LON_COL = "longitude"
LAT_COL = "latitude"
STATUS_COL = "status"
TRAJ_COL = "trajectory"
TIME_COL = "time"

# Coordinate Reference Systems
SOURCE_CRS = "EPSG:4326"   # Input coordinate system (WGS84 lat/lon)
TARGET_CRS = "EPSG:32633"  # Target metric coordinate system (UTM Zone 33N)

FILE_PATTERN = "*.csv"

# Fixed bounding box in target CRS units (meters); set to None to auto-fit
MANUAL_BOUNDS = {
    "xmin": 410000, "xmax": 480000,
    "ymin": 3945000, "ymax": 4010000,
}

MASK_ZERO_AS_NODATA = True  # Mask non-beaching cells to NoData (NaN) for raster transparency
KDE_BANDWIDTH_METERS = 500  # Gaussian smoothing radius in meters

SEASONS = ["winter", "spring", "summer", "autumn"]


def season_from_folder(folder_name: str):
    """Extract season name from folder prefix (e.g., 'autumn_2025-11-21' -> 'autumn')."""
    season = folder_name.split("_")[0].strip().lower()
    return season if season in SEASONS else None


def first_beaching_points(csv_path: str):
    """Extract total trajectory count and the first beaching event coordinates per trajectory."""
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

    # Select earliest beaching timestamp per trajectory
    beached = beached.sort_values([TRAJ_COL, TIME_COL])
    first_beach = beached.drop_duplicates(subset=TRAJ_COL, keep="first")

    return n_particles, first_beach[LON_COL].values, first_beach[LAT_COL].values


def collect_season_data(input_root: str):
    """Iterate through dataset folders and aggregate initial beaching coordinates by season."""
    season_data = {s: {"n_particles": 0, "lon": [], "lat": []} for s in SEASONS}

    day_folders = sorted(
        d for d in os.listdir(input_root)
        if os.path.isdir(os.path.join(input_root, d))
    )
    if not day_folders:
        raise FileNotFoundError(f"No subfolders found in {input_root}")

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
        print(f"Skipping {len(skipped_folders)} folder(s) with unrecognized season prefix.")

    print(f"Found {len(work_items)} CSV file(s) across {len(day_folders) - len(skipped_folders)} folder(s). Starting...\n")

    beaching_events_so_far = 0
    progress = tqdm(work_items, desc="Processing runs", unit="file")
    for season, folder, csv_path in progress:
        n_particles, lon, lat = first_beaching_points(csv_path)
        season_data[season]["n_particles"] += n_particles
        if len(lon):
            season_data[season]["lon"].append(lon)
            season_data[season]["lat"].append(lat)
            beaching_events_so_far += len(lon)

        progress.set_postfix({
            "folder": folder,
            "particles": sum(d["n_particles"] for d in season_data.values()),
            "beached": beaching_events_so_far,
        })

    return season_data


def _bin_beaching_counts(lon: np.ndarray, lat: np.ndarray, resolution=RESOLUTION):
    """Project geographic points to metric CRS and aggregate counts into 2D grid cells."""
    transformer = Transformer.from_crs(SOURCE_CRS, TARGET_CRS, always_xy=True)
    x, y = transformer.transform(lon, lat)

    if MANUAL_BOUNDS:
        xmin, xmax = MANUAL_BOUNDS["xmin"], MANUAL_BOUNDS["xmax"]
        ymin, ymax = MANUAL_BOUNDS["ymin"], MANUAL_BOUNDS["ymax"]
    else:
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
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
    # Transpose and invert vertical axis to align with standard raster orientation (North on top)
    count = count.T[::-1, :]

    transform = from_origin(xmin, ymax, resolution, resolution)
    return count, transform


def make_probability_grid(lon: np.ndarray, lat: np.ndarray, n_particles: int, resolution=RESOLUTION):
    """Compute raw beaching probability per cell (beached counts divided by total particles)."""
    count, transform = _bin_beaching_counts(lon, lat, resolution)
    probability = (count / n_particles).astype("float32") if n_particles > 0 else count.astype("float32")

    if MASK_ZERO_AS_NODATA:
        probability[probability == 0] = np.nan

    return probability, transform


def make_density_grid(lon: np.ndarray, lat: np.ndarray, n_particles: int, resolution=RESOLUTION):
    """Generate continuous KDE density map by applying Gaussian smoothing to the probability grid."""
    count, transform = _bin_beaching_counts(lon, lat, resolution)
    probability = (count / n_particles).astype("float32") if n_particles > 0 else count.astype("float32")
    probability = np.nan_to_num(probability, nan=0.0)

    sigma_cells = max(KDE_BANDWIDTH_METERS / resolution, 0.01)
    density = gaussian_filter(probability, sigma=sigma_cells, mode="constant", cval=0.0)

    return density.astype("float32"), transform


def save_geotiff(grid: np.ndarray, transform, out_path: str, nodata=None):
    """Export 2D array as a single-band GeoTIFF raster."""
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
            tqdm.write(f"{season}: {n_particles} particles simulated, none beached. Skipping grid.")
            continue

        lon = np.concatenate(data["lon"])
        lat = np.concatenate(data["lat"])

        tqdm.write(f"{season}: {n_particles} particles simulated, {len(lon)} beaching events -> gridding...")

        grid, transform = make_probability_grid(lon, lat, n_particles)
        out_path = os.path.join(OUTPUT_DIR, f"{season}_beaching_probability_100m.tif")
        save_geotiff(grid, transform, out_path, nodata=(np.nan if MASK_ZERO_AS_NODATA else None))

        density, density_transform = make_density_grid(lon, lat, n_particles)
        density_path = os.path.join(OUTPUT_DIR, f"{season}_beaching_density_100m.tif")
        save_geotiff(density, density_transform, density_path, nodata=None)

    # Combined all-seasons dataset processing
    print()
    all_lon_parts = [l for s in SEASONS for l in season_data[s]["lon"]]
    all_particles = sum(season_data[s]["n_particles"] for s in SEASONS)

    if all_particles == 0 or not all_lon_parts:
        print("No data available to build the combined all-seasons grid.")
    else:
        all_lat_parts = [l for s in SEASONS for l in season_data[s]["lat"]]
        combined_lon = np.concatenate(all_lon_parts)
        combined_lat = np.concatenate(all_lat_parts)

        print(f"Combined (all seasons): {all_particles} particles simulated, {len(combined_lon)} beaching events -> gridding...")

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