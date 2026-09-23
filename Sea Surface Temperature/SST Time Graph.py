# pip install pandas xarray netCDF4 matplotlib openpyxl

import glob
import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# File & Output Directories
DRIFTER_EXCEL_PATH = r"E:\University\Applied Oceanography\Dissertation\Data\Drifter Data\Drifter 1.csv"
COPERNICUS_DATA_PATH = r"E:\University\Applied Oceanography\Dissertation\Data\SST\SST.nc"
OUTPUT_DIR = r"E:\University\Applied Oceanography\Dissertation\Results\Sea Surface Temperature"

# Configuration Options
COPERNICUS_FILE_PATTERN = "*.nc"
SST_VAR_CANDIDATES = ["analysed_sst", "thetao", "sst", "sea_surface_temperature"]


def load_drifter_data(path: str) -> pd.DataFrame:
    """Load and validate drifter CSV data with UTC timestamp sorting."""
    df = pd.read_csv(path)
    required_cols = {"Latitude", "Longitude", "SST", "UtcTimestamp"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in CSV: {missing}")

    df["UtcTimestamp"] = pd.to_datetime(df["UtcTimestamp"], utc=True)
    df = df.sort_values("UtcTimestamp").reset_index(drop=True)
    return df


def load_copernicus_dataset(path: str) -> xr.Dataset:
    """Load single or multi-file Copernicus NetCDF datasets."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, COPERNICUS_FILE_PATTERN)))
        if not files:
            raise FileNotFoundError(f"No files matching {COPERNICUS_FILE_PATTERN} in {path}")
        ds = xr.open_mfdataset(files, combine="by_coords")
    else:
        ds = xr.open_dataset(path)
    return ds


def find_sst_variable(ds: xr.Dataset) -> str:
    """Identify the SST variable name within the NetCDF dataset."""
    for name in SST_VAR_CANDIDATES:
        if name in ds.data_vars:
            return name
    raise ValueError(
        f"Could not automatically identify SST variable in dataset variables: {list(ds.data_vars)}"
    )


def extract_matching_sst(ds: xr.Dataset, sst_var: str, df: pd.DataFrame) -> pd.DataFrame:
    """Extract nearest-neighbor Copernicus SST values corresponding to drifter time and location."""
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    time_name = "time" if "time" in ds.coords else "time_counter"

    # Align dataset coordinate timezone to UTC naive for direct timestamp matching
    ds_time = pd.to_datetime(ds[time_name].values)
    if ds_time.tz is not None:
        ds = ds.assign_coords({time_name: ds_time.tz_localize(None)})

    values = []
    for _, row in df.iterrows():
        try:
            point = ds[sst_var].sel(
                {
                    lat_name: row["Latitude"],
                    lon_name: row["Longitude"],
                    time_name: row["UtcTimestamp"].tz_localize(None),
                },
                method="nearest",
            )
            val = float(point.values)
            # Convert Kelvin to Celsius if temperature values are above 100
            if val > 100:
                val -= 273.15
        except Exception:
            val = np.nan
        values.append(val)

    df = df.copy()
    df["Copernicus_SST"] = values
    return df


def plot_time_series(df: pd.DataFrame, output_dir: str, filename="sst_timeseries_comparison.png") -> str:
    """Plot time series comparison of in-situ drifter SST vs gridded Copernicus SST."""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["UtcTimestamp"], df["SST"], marker="o", markersize=3, linewidth=1, label="Drifter SST")
    ax.plot(df["UtcTimestamp"], df["Copernicus_SST"], marker="x", markersize=3, linewidth=1, label="Copernicus SST")

    ax.set_xlabel("UTC Time")
    ax.set_ylabel("SST (°C)")
    ax.set_title("Drifter vs Copernicus Sea Surface Temperature")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    fig.autofmt_xdate()

    out_path = os.path.join(output_dir, filename)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Plot saved to: {out_path}")
    return out_path


def main():
    df = load_drifter_data(DRIFTER_EXCEL_PATH)
    ds = load_copernicus_dataset(COPERNICUS_DATA_PATH)
    sst_var = find_sst_variable(ds)
    print(f"Using Copernicus SST variable: '{sst_var}'")

    df = extract_matching_sst(ds, sst_var, df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    merged_csv = os.path.join(OUTPUT_DIR, "sst_comparison_data.csv")
    df.to_csv(merged_csv, index=False)
    print(f"Merged data saved to: {merged_csv}")

    plot_time_series(df, OUTPUT_DIR)


if __name__ == "__main__":
    main()