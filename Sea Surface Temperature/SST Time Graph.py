"""
Compare in-situ surface-drifter SST measurements against Copernicus Marine
Service gridded SST data and plot a time series.

Requirements (install with pip):
    pip install pandas xarray netCDF4 matplotlib openpyxl

Author: auto-generated for user
"""

import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ----------------------------------------------------------------------
# 1. USER CONFIGURATION -- edit these paths for your machine
# ----------------------------------------------------------------------
DRIFTER_EXCEL_PATH = r"E:\University\Applied Oceanography\Dissertation\Data\Drifter Data\Drifter 1.csv"     # your Excel file
COPERNICUS_DATA_PATH = r"E:\University\Applied Oceanography\Dissertation\Data\SST\SST.nc"          # folder OR single .nc file
OUTPUT_DIR = r"E:\University\Applied Oceanography\Dissertation\Results\Sea Surface Temperature"                  # where results are saved

# If your Copernicus files are split (e.g. one .nc per day), point
# COPERNICUS_DATA_PATH at the folder and set this pattern:
COPERNICUS_FILE_PATTERN = "*.nc"

# Candidate variable names for SST -- the first one found in the
# dataset is used automatically. Add to this list if none match.
SST_VAR_CANDIDATES = ["analysed_sst", "thetao", "sst", "sea_surface_temperature"]


def load_drifter_data(path):
    df = pd.read_csv(path)
    required_cols = {"Latitude", "Longitude", "SST", "UtcTimestamp"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Excel file is missing expected columns: {missing}")

    df["UtcTimestamp"] = pd.to_datetime(df["UtcTimestamp"], utc=True)
    df = df.sort_values("UtcTimestamp").reset_index(drop=True)
    return df


def load_copernicus_dataset(path):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, COPERNICUS_FILE_PATTERN)))
        if not files:
            raise FileNotFoundError(
                f"No files matching {COPERNICUS_FILE_PATTERN} in {path}"
            )
        ds = xr.open_mfdataset(files, combine="by_coords")
    else:
        ds = xr.open_dataset(path)
    return ds


def find_sst_variable(ds):
    for name in SST_VAR_CANDIDATES:
        if name in ds.data_vars:
            return name
    raise ValueError(
        f"Could not find an SST variable automatically. "
        f"Available variables: {list(ds.data_vars)}. "
        f"Add the correct name to SST_VAR_CANDIDATES."
    )


def extract_matching_sst(ds, sst_var, df):
    """For every drifter observation, pull the nearest-in-space/time
    Copernicus SST value."""

    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    time_name = "time" if "time" in ds.coords else "time_counter"

    # Make the dataset's time coordinate timezone-naive UTC so it can
    # be compared directly against UtcTimestamp.
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
            # Convert from Kelvin to Celsius if needed
            if val > 100:
                val -= 273.15
        except Exception:
            val = np.nan
        values.append(val)

    df = df.copy()
    df["Copernicus_SST"] = values
    return df


def plot_time_series(df, output_dir, filename="sst_timeseries_comparison.png"):
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["UtcTimestamp"], df["SST"], marker="o", markersize=3,
            linewidth=1, label="Drifter SST")
    ax.plot(df["UtcTimestamp"], df["Copernicus_SST"], marker="x", markersize=3,
            linewidth=1, label="Copernicus SST")

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