import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import openmeteo_requests
from opendrift.models.oceandrift import OceanDrift
from opendrift.readers import reader_netCDF_CF_generic
import pandas as pd
import requests_cache
from retry_requests import retry
import xarray as xr

# Constants
EARTH_RADIUS_M = 6371000.0

# Input Files & Directories
DRIFTER_CSV = r"E:\University\Applied Oceanography\Dissertation\Data\Drifter Data\Drifter 2.csv"
CURRENTS_FILE = r"E:\University\Applied Oceanography\Dissertation\Data\Currents\MonthCurrentsAnalysis.nc"
WAVE_FILE = r"E:\University\Applied Oceanography\Dissertation\Data\Waves\MonthWaveAnalysis.nc"

# Output Directory
OUTPUT_DIR = r"E:\University\Applied Oceanography\Dissertation\Results\OpenOil\Drifter 2\1 Week Cycle"

# Simulation Parameters
FID_START = 0
SIMULATION_DURATION_DAYS = 7
MODEL_TIME_STEP_SECONDS = 900
OUTPUT_EVERY_SECONDS = 1800
USE_WAVE_STOKES_DRIFT = True

# Open-Meteo Configuration
OPENMETEO_MODEL = "italia_meteo_arpae_icon_2i"
WIND_GRID_MARGIN_DEG = 0.02


def load_drifter_track(
    path: str, fid_start: int, duration_days: int
) -> tuple[pd.DataFrame, pd.DataFrame, float, float, pd.Timestamp, pd.Timestamp]:
    """Load, clean, and extract the seed position and track window for a given drifter FID."""
    if os.path.isdir(path):
        csv_files = sorted(glob.glob(os.path.join(path, "*.csv")))
    else:
        csv_files = [path] if os.path.isfile(path) else []

    if not csv_files:
        raise FileNotFoundError(f"Target drifter CSV file or folder path not found: {path}")

    frames = [pd.read_csv(f)[["FID", "UtcTimestamp", "Latitude", "Longitude"]] for f in csv_files]
    drifter_df = pd.concat(frames, ignore_index=True)
    drifter_df["time_utc"] = pd.to_datetime(drifter_df["UtcTimestamp"], utc=True).dt.tz_localize(None)

    drifter_df = (
        drifter_df.dropna(subset=["time_utc"])
        .sort_values("time_utc")
        .drop_duplicates(subset="time_utc")
        .reset_index(drop=True)
        .rename(columns={"Latitude": "lat", "Longitude": "lon"})
    )

    if drifter_df.empty:
        raise ValueError(f"Target drifter CSV resulted in an empty dataset: {path}")

    start_rows = drifter_df.loc[drifter_df["FID"] == fid_start]
    if start_rows.empty:
        raise ValueError(f"FID_START {fid_start} not found in the processed drifter data.")

    first_point = start_rows.iloc[0]
    start_lon = float(first_point["lon"])
    start_lat = float(first_point["lat"])
    start_time = first_point["time_utc"]

    target_end_time = start_time + pd.Timedelta(days=duration_days)
    available_end_time = drifter_df["time_utc"].max()
    end_time = pd.Timestamp(min(target_end_time, available_end_time))

    if end_time <= start_time:
        raise ValueError(
            f"No drifter data available after start time ({start_time}) for FID_START {fid_start}."
        )

    actual_track_df = drifter_df.loc[
        (drifter_df["time_utc"] >= start_time) & (drifter_df["time_utc"] <= end_time)
    ].reset_index(drop=True)

    if actual_track_df.empty:
        raise ValueError(
            f"No FID_START ({fid_start}) observations fall within [{start_time}, {end_time}]."
        )

    return drifter_df, actual_track_df, start_lon, start_lat, start_time, end_time


def fetch_openmeteo_wind_dataset(
    start_lat: float,
    start_lon: float,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    actual_track_df: pd.DataFrame,
) -> xr.Dataset:
    """Query Open-Meteo historical forecast API and construct a spatial wind Dataset."""
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": start_lat,
        "longitude": start_lon,
        "hourly": ["wind_speed_10m", "wind_direction_10m"],
        "models": OPENMETEO_MODEL,
        "start_date": start_time.strftime("%Y-%m-%d"),
        "end_date": end_time.strftime("%Y-%m-%d"),
    }

    responses = openmeteo.weather_api(url, params=params)
    hourly = responses[0].Hourly()

    hourly_wind_speed = hourly.Variables(0).ValuesAsNumpy()
    hourly_wind_direction = hourly.Variables(1).ValuesAsNumpy()

    hourly_time = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    ).tz_localize(None)

    direction_rad = np.radians(hourly_wind_direction)
    u10_1d = -hourly_wind_speed * np.sin(direction_rad)
    v10_1d = -hourly_wind_speed * np.cos(direction_rad)

    lat_min = actual_track_df["lat"].min() - WIND_GRID_MARGIN_DEG
    lat_max = actual_track_df["lat"].max() + WIND_GRID_MARGIN_DEG
    lon_min = actual_track_df["lon"].min() - WIND_GRID_MARGIN_DEG
    lon_max = actual_track_df["lon"].max() + WIND_GRID_MARGIN_DEG

    grid_lats = np.array([lat_min, (lat_min + lat_max) / 2.0, lat_max])
    grid_lons = np.array([lon_min, (lon_min + lon_max) / 2.0, lon_max])

    u10_3d = np.broadcast_to(u10_1d[:, None, None], (len(hourly_time), 3, 3)).copy()
    v10_3d = np.broadcast_to(v10_1d[:, None, None], (len(hourly_time), 3, 3)).copy()

    wind_dataset = xr.Dataset(
        data_vars={
            "u10": (["time", "latitude", "longitude"], u10_3d),
            "v10": (["time", "latitude", "longitude"], v10_3d),
        },
        coords={
            "time": hourly_time,
            "latitude": grid_lats,
            "longitude": grid_lons,
        },
    )
    wind_dataset["u10"].attrs["standard_name"] = "eastward_wind"
    wind_dataset["v10"].attrs["standard_name"] = "northward_wind"

    return wind_dataset


def run_opendrift_simulation(
    start_lon: float,
    start_lat: float,
    start_time: pd.Timestamp,
    simulation_duration: pd.Timedelta,
    wind_dataset: xr.Dataset,
) -> pd.DataFrame:
    """Execute OceanDrift simulation and return predicted trajectory DataFrame."""
    wind_reader = reader_netCDF_CF_generic.Reader(
        wind_dataset, name="Open-Meteo historical forecast wind"
    )
    wave_reader = reader_netCDF_CF_generic.Reader(WAVE_FILE, name="Copernicus waves")
    current_reader = reader_netCDF_CF_generic.Reader(CURRENTS_FILE, name="Copernicus surface currents")

    model = OceanDrift(loglevel=20)
    model.add_reader([wind_reader, wave_reader, current_reader])

    model.set_config("drift:stokes_drift", USE_WAVE_STOKES_DRIFT)
    model.set_config("drift:use_tabularised_stokes_drift", False)

    model.seed_elements(
        lon=start_lon,
        lat=start_lat,
        number=1,
        radius=0,
        time=start_time.to_pydatetime(),
        z=0,
    )

    model.run(
        duration=simulation_duration,
        time_step=MODEL_TIME_STEP_SECONDS,
        time_step_output=OUTPUT_EVERY_SECONDS,
    )

    predicted_times = pd.to_datetime(model.result.time.values)
    predicted_lons = model.result.lon.values[0, :]
    predicted_lats = model.result.lat.values[0, :]

    return (
        pd.DataFrame(
            {
                "time_utc": predicted_times,
                "pred_lat": predicted_lats,
                "pred_lon": predicted_lons,
            }
        )
        .sort_values("time_utc")
        .reset_index(drop=True)
    )


def compute_separation_distance(
    actual_track_df: pd.DataFrame, predicted_df: pd.DataFrame
) -> pd.DataFrame:
    """Calculate Haversine separation distance (km) between observed and predicted trajectories."""
    df_actual = actual_track_df.copy()
    df_pred = predicted_df.copy()
    df_actual["time_utc"] = df_actual["time_utc"].astype("datetime64[ns]")
    df_pred["time_utc"] = df_pred["time_utc"].astype("datetime64[ns]")

    merged_df = pd.merge_asof(
        df_actual.sort_values("time_utc"),
        df_pred.sort_values("time_utc"),
        on="time_utc",
        direction="nearest",
        tolerance=pd.Timedelta("30min"),
    ).dropna(subset=["pred_lat", "pred_lon"]).reset_index(drop=True)

    lat1, lon1, lat2, lon2 = map(
        np.radians,
        [merged_df["lat"], merged_df["lon"], merged_df["pred_lat"], merged_df["pred_lon"]],
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    haversine_array = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    merged_df["separation_km"] = (2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(haversine_array))) / 1000.0

    return merged_df


def save_results_and_plots(
    actual_track_df: pd.DataFrame,
    predicted_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> None:
    """Save trajectory CSVs, trajectory plots, separation distance plots, and summary text."""
    actual_track_df.to_csv(os.path.join(OUTPUT_DIR, "drifter_actual_track.csv"), index=False)
    predicted_df.to_csv(os.path.join(OUTPUT_DIR, "predicted_track.csv"), index=False)
    merged_df.to_csv(os.path.join(OUTPUT_DIR, "separation_distances.csv"), index=False)

    start_str = start_time.strftime("%Y-%m-%d %H:%M UTC")
    end_str = end_time.strftime("%Y-%m-%d %H:%M UTC")

    # Trajectory comparison plot
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.plot(
        actual_track_df["lon"],
        actual_track_df["lat"],
        "-o",
        color="blue",
        label="Actual drifter track",
        markersize=3,
        linewidth=1.5,
        zorder=3,
    )
    ax.plot(
        predicted_df["pred_lon"],
        predicted_df["pred_lat"],
        "-o",
        color="red",
        label="OpenDrift track",
        markersize=3,
        linewidth=1.5,
        zorder=2,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"Drifter Trajectory Validation: Actual vs. OpenDrift\nStart: {start_str}   |   End: {end_str}"
    )
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "track_comparison.png"), dpi=200)
    plt.close(fig)

    # Separation distance plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(merged_df["time_utc"], merged_df["separation_km"], "-o", color="black", markersize=3)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Separation distance (km)")
    ax.set_title(
        f"Distance Between Observed and OpenDrift Positions\nStart: {start_str}   |   End: {end_str}"
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "separation_distance.png"), dpi=200)
    plt.close(fig)

    # Summary report
    summary_path = os.path.join(OUTPUT_DIR, "separation_error_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Drifter Trajectory Validation Summary\n")
        f.write(f"Start: {start_str}\n")
        f.write(f"End:   {end_str}\n\n")
        f.write(f"Mean separation error:  {merged_df['separation_km'].mean():.3f} km\n")
        f.write(f"Max separation error:   {merged_df['separation_km'].max():.3f} km\n")
        f.write(f"Final separation error: {merged_df['separation_km'].iloc[-1]:.3f} km\n")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    drifter_df, actual_track_df, start_lon, start_lat, start_time, end_time = load_drifter_track(
        DRIFTER_CSV, FID_START, SIMULATION_DURATION_DAYS
    )

    simulation_duration = end_time - start_time
    print(f"Drifter data loaded. Observations: {len(drifter_df)}")
    print(f"Start: {start_time} (Lat {start_lat}, Lon {start_lon})")
    print(f"End:   {end_time} (Duration: {simulation_duration})")

    print("\nFetching historical forecast wind data from Open-Meteo...")
    wind_dataset = fetch_openmeteo_wind_dataset(
        start_lat, start_lon, start_time, end_time, actual_track_df
    )

    print("\nRunning OpenDrift trajectory model...")
    predicted_df = run_opendrift_simulation(
        start_lon, start_lat, start_time, simulation_duration, wind_dataset
    )

    merged_df = compute_separation_distance(actual_track_df, predicted_df)

    save_results_and_plots(actual_track_df, predicted_df, merged_df, start_time, end_time)

    print(f"\nMean separation error:  {merged_df['separation_km'].mean():.3f} km")
    print(f"Max separation error:   {merged_df['separation_km'].max():.3f} km")
    print(f"Final separation error: {merged_df['separation_km'].iloc[-1]:.3f} km")
    print(f"\nAll outputs successfully saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()