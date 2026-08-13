# LOADING LIBRARIES
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import openmeteo_requests
import requests_cache
from retry_requests import retry
from opendrift.models.oceandrift import OceanDrift
from opendrift.readers import reader_netCDF_CF_generic

# Constants
EARTH_RADIUS_M = 6371000.0

# ----------------------------------------------------------------------

# INPUT FILES & DIRECTORIES (same for every day)
DRIFTER_CSV = r"E:\University\Applied Oceanography\Dissertation\Data\Drifter Data\Drifter 1.csv"
CURRENTS_FILE = r"E:\University\Applied Oceanography\Dissertation\Data\Currents\MonthCurrentsAnalysis.nc"
WAVE_FILE = r"E:\University\Applied Oceanography\Dissertation\Data\Waves\MonthWaveAnalysis.nc"

# BASE OUTPUT DIRECTORY - must contain subfolders "Day 1", "Day 2", ..., "Day 32"
OUTPUT_BASE_DIR = r"E:\University\Applied Oceanography\Dissertation\Results\OpenOil\Drifter 1\1 Day Cycle"

# EXCEL SCHEDULE - maps each Day number to the FID to seed from
EXCEL_PATH = r"E:\University\Applied Oceanography\Dissertation\Data\Drifter Day FID\Drifter 1 Number Log.xlsx"
DAY_COLUMN = "Day"
FID_COLUMN = "FID"

# ----------------------------------------------------------------------

# CONFIGURATION PARAMETERS
SIMULATION_DURATION_DAYS = 1  # simulation automatically ends this many days after the start time
MODEL_TIME_STEP_SECONDS = 900
OUTPUT_EVERY_SECONDS = 1800
USE_WAVE_STOKES_DRIFT = True

# Open-Meteo historical forecast configuration
OPENMETEO_MODEL = "italia_meteo_arpae_icon_2i"
WIND_GRID_MARGIN_DEG = 0.02

# ----------------------------------------------------------------------

# LOAD DRIFTER DATA ONCE (shared across all days)
if os.path.isdir(DRIFTER_CSV):
    csv_files = sorted(glob.glob(os.path.join(DRIFTER_CSV, "*.csv")))
else:
    csv_files = [DRIFTER_CSV] if os.path.isfile(DRIFTER_CSV) else []

if not csv_files:
    raise FileNotFoundError(f"Target drifter CSV file or folder path not found: {DRIFTER_CSV}")

frames = []
for f in csv_files:
    df = pd.read_csv(f)
    frames.append(df[["FID", "UtcTimestamp", "Latitude", "Longitude"]])

drifter_df = pd.concat(frames, ignore_index=True)
drifter_df["time_utc"] = pd.to_datetime(drifter_df["UtcTimestamp"], utc=True).dt.tz_localize(None)

drifter_df = (
    drifter_df.dropna(subset=["time_utc"])
    .sort_values("time_utc")
    .drop_duplicates(subset="time_utc")
    .reset_index(drop=True)
)
drifter_df = drifter_df.rename(columns={"Latitude": "lat", "Longitude": "lon"})

if drifter_df.empty:
    raise ValueError(f"Target drifter CSV resulted in an empty dataset: {DRIFTER_CSV}")

print(f"Drifter data loaded. Observations: {len(drifter_df)}")

# ----------------------------------------------------------------------

# SHARED OPEN-METEO SESSION (cache persists across all days in the loop)
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# ----------------------------------------------------------------------

def run_simulation_for_fid(fid_start, output_dir):
    """Runs the full OceanDrift pipeline for one FID_START, saving outputs to output_dir."""

    os.makedirs(output_dir, exist_ok=True)

    start_rows = drifter_df.loc[drifter_df["FID"] == fid_start]
    if start_rows.empty:
        raise ValueError(f"FID_START {fid_start} not found in the processed drifter data.")
    first_point = start_rows.iloc[0]

    start_lon = first_point["lon"]
    start_lat = first_point["lat"]
    pd_start_time = first_point["time_utc"]
    dynamic_start_time_py = pd_start_time.to_pydatetime()

    target_end_time = pd_start_time + pd.Timedelta(days=SIMULATION_DURATION_DAYS)
    available_end_time = drifter_df["time_utc"].max()
    end_time_utc = pd.Timestamp(min(target_end_time, available_end_time))

    if end_time_utc <= pd_start_time:
        raise ValueError(
            f"No drifter data available after the FID_START ({fid_start}) start time "
            f"({pd_start_time}). Check your drifter CSV coverage."
        )

    print(f"  DYNAMIC SEEDING matching FID_START {fid_start}:")
    print(f"    Time (UTC): {pd_start_time}")
    print(f"    Position:   Lat {start_lat}, Lon {start_lon}")
    print(f"  SIMULATION END (start + {SIMULATION_DURATION_DAYS} days, capped to available data):")
    print(f"    Time (UTC): {end_time_utc}")
    if target_end_time > available_end_time:
        print(f"    NOTE: requested {SIMULATION_DURATION_DAYS}-day end ({target_end_time}) exceeds "
              f"available drifter data ({available_end_time}); simulation end was capped.")

    simulation_duration_delta = end_time_utc - pd_start_time
    print(f"  Simulation duration: {simulation_duration_delta}")

    # Actual track used for comparison/plots
    actual_track_df = drifter_df.loc[
        (drifter_df["time_utc"] >= pd_start_time)
        & (drifter_df["time_utc"] <= end_time_utc)
    ].reset_index(drop=True)

    if actual_track_df.empty:
        raise ValueError(
            f"No FID_START ({fid_start}) observations fall within the "
            f"[{pd_start_time}, {end_time_utc}] window."
        )

    print(f"  Actual track points in simulation window: {len(actual_track_df)}")

    # --------------------------------------------------------------

    # FETCHING WIND DATA FROM OPEN-METEO HISTORICAL FORECAST API
    print("  Fetching historical forecast wind data from Open-Meteo...")

    url = "https://api.open-meteo.com/v1/forecast"
    openmeteo_params = {
        "latitude": start_lat,
        "longitude": start_lon,
        "hourly": ["wind_speed_10m", "wind_direction_10m"],
        "models": OPENMETEO_MODEL,
        "start_date": pd_start_time.strftime("%Y-%m-%d"),
        "end_date": end_time_utc.strftime("%Y-%m-%d"),
    }

    responses = openmeteo.weather_api(url, params=openmeteo_params)
    response = responses[0]

    hourly = response.Hourly()
    hourly_wind_speed_10m = hourly.Variables(0).ValuesAsNumpy()
    hourly_wind_direction_10m = hourly.Variables(1).ValuesAsNumpy()

    hourly_time = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    ).tz_localize(None)

    direction_rad = np.radians(hourly_wind_direction_10m)
    u10_1d = -hourly_wind_speed_10m * np.sin(direction_rad)
    v10_1d = -hourly_wind_speed_10m * np.cos(direction_rad)

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

    print(f"  Open-Meteo wind data retrieved: {len(hourly_time)} hourly steps "
          f"({hourly_time[0]} to {hourly_time[-1]})")

    # --------------------------------------------------------------

    # READING ENVIRONMENT FORECAST FILES (re-created fresh each call - readers are stateful)
    wind_reader = reader_netCDF_CF_generic.Reader(wind_dataset, name="Open-Meteo historical forecast wind")
    wave_reader = reader_netCDF_CF_generic.Reader(WAVE_FILE, name="Copernicus waves")
    current_reader = reader_netCDF_CF_generic.Reader(CURRENTS_FILE, name="Copernicus surface currents")

    # --------------------------------------------------------------

    # RUNNING OPENDRIFT
    model = OceanDrift(loglevel=20)
    model.add_reader([wind_reader, wave_reader, current_reader])

    model.set_config("drift:stokes_drift", USE_WAVE_STOKES_DRIFT)
    model.set_config("drift:use_tabularised_stokes_drift", False)

    model.seed_elements(
        lon=start_lon,
        lat=start_lat,
        number=1,
        radius=0,
        time=dynamic_start_time_py,
        z=0
    )

    model.run(
        duration=simulation_duration_delta,
        time_step=MODEL_TIME_STEP_SECONDS,
        time_step_output=OUTPUT_EVERY_SECONDS,
    )

    # --------------------------------------------------------------

    # EXTRACTING TRACKS & PROCESSING SEPARATION METRICS
    predicted_times = pd.to_datetime(model.result.time.values)
    predicted_lons = model.result.lon.values[0, :]
    predicted_lats = model.result.lat.values[0, :]

    predicted_df = pd.DataFrame({
        "time_utc": predicted_times,
        "pred_lat": predicted_lats,
        "pred_lon": predicted_lons
    }).sort_values("time_utc").reset_index(drop=True)

    df_actual_calc = actual_track_df.copy()
    df_pred_calc = predicted_df.copy()
    df_actual_calc["time_utc"] = df_actual_calc["time_utc"].astype("datetime64[ns]")
    df_pred_calc["time_utc"] = df_pred_calc["time_utc"].astype("datetime64[ns]")

    merged_df = pd.merge_asof(
        df_actual_calc.sort_values("time_utc"),
        df_pred_calc.sort_values("time_utc"),
        on="time_utc",
        direction="nearest",
        tolerance=pd.Timedelta("30min"),
    ).dropna(subset=["pred_lat", "pred_lon"]).reset_index(drop=True)

    lat1, lon1, lat2, lon2 = map(np.radians,
                                 [merged_df["lat"], merged_df["lon"], merged_df["pred_lat"], merged_df["pred_lon"]])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    haversine_array = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    merged_df["separation_km"] = (2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(haversine_array))) / 1000.0

    # --------------------------------------------------------------

    # SAVING OUTPUT DATA & PLOTS
    actual_track_df.to_csv(os.path.join(output_dir, "drifter_actual_track.csv"), index=False)
    predicted_df.to_csv(os.path.join(output_dir, "predicted_track.csv"), index=False)
    merged_df.to_csv(os.path.join(output_dir, "separation_distances.csv"), index=False)

    start_str = pd_start_time.strftime("%Y-%m-%d %H:%M UTC")
    end_str = end_time_utc.strftime("%Y-%m-%d %H:%M UTC")

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.plot(actual_track_df["lon"], actual_track_df["lat"], "-o", color="blue", label="Actual drifter track",
            markersize=3, linewidth=1.5, zorder=3)
    ax.plot(predicted_df["pred_lon"], predicted_df["pred_lat"], "-o", color="red", label="OpenDrift track",
            markersize=3, linewidth=1.5, zorder=2)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"Drifter Trajectory Validation: Actual vs. OpenDrift\n"
        f"Start: {start_str}   |   End: {end_str}"
    )
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "track_comparison.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(merged_df["time_utc"], merged_df["separation_km"], "-o", color="black", markersize=3)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Separation distance (km)")
    ax.set_title(
        f"Distance Between Observed and OpenDrift Positions\n"
        f"Start: {start_str}   |   End: {end_str}"
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "separation_distance.png"), dpi=200)
    plt.close(fig)

    mean_err = merged_df['separation_km'].mean()
    max_err = merged_df['separation_km'].max()
    final_err = merged_df['separation_km'].iloc[-1]

    summary_path = os.path.join(output_dir, "separation_error_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Drifter Trajectory Validation Summary\n")
        f.write(f"FID_START: {fid_start}\n")
        f.write(f"Start: {start_str}\n")
        f.write(f"End:   {end_str}\n\n")
        f.write(f"Mean separation error:  {mean_err:.3f} km\n")
        f.write(f"Max separation error:   {max_err:.3f} km\n")
        f.write(f"Final separation error: {final_err:.3f} km\n")

    print(f"  Mean separation error:  {mean_err:.3f} km")
    print(f"  Max separation error:   {max_err:.3f} km")
    print(f"  Final separation error: {final_err:.3f} km")
    print(f"  Outputs saved to: {output_dir}")


# ----------------------------------------------------------------------

# LOOP OVER EVERY DAY IN THE EXCEL SCHEDULE

schedule_df = pd.read_excel(EXCEL_PATH)  # use pd.read_csv(EXCEL_PATH) instead if it's a .csv

for _, row in schedule_df.sort_values(DAY_COLUMN).iterrows():
    day_num = int(row[DAY_COLUMN])
    fid = row[FID_COLUMN]
    day_output_dir = os.path.join(OUTPUT_BASE_DIR, f"Day {day_num}")

    print(f"\n{'=' * 70}")
    print(f"RUNNING DAY {day_num}  (FID {fid})")
    print(f"{'=' * 70}")

    try:
        run_simulation_for_fid(fid, day_output_dir)
    except Exception as e:
        print(f"  ERROR on Day {day_num} (FID {fid}): {e}")
        continue

print("\nAll days processed.")