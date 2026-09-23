# pip install matplotlib numpy pandas xarray openmeteo-requests requests-cache retry-requests opendrift openpyxl

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
from opendrift.models.plastdrift import PlastDrift
from opendrift.readers import reader_netCDF_CF_generic

# Earth radius (metres), used for haversine distance calculations
EARTH_RADIUS_M = 6371000.0

# ============================================================================
# INPUTS
# ============================================================================

# Drifter GPS track: a single CSV file or a folder of CSVs, each with
# columns FID, UtcTimestamp, Latitude, Longitude
DRIFTER_CSV = r"E:\University\Applied Oceanography\Dissertation\Data\Drifter Data\Drifter 1.csv"

# NetCDF file of ocean surface currents (Copernicus), used as a model reader
CURRENTS_FILE = r"E:\University\Applied Oceanography\Dissertation\Data\Currents\MonthCurrentsAnalysis.nc"

# NetCDF file of wave data (Copernicus), used as a model reader for Stokes drift
WAVE_FILE = r"E:\University\Applied Oceanography\Dissertation\Data\Waves\MonthWaveAnalysis.nc"

# Excel file mapping each simulation "Day" to the drifter FID it should start from.
# Must contain the columns named in DAY_COLUMN and FID_COLUMN below.
EXCEL_PATH = r"E:\University\Applied Oceanography\Dissertation\Data\Drifter Day FID\Drifter 1 Number Log.xlsx"
DAY_COLUMN = "Day"   # column in EXCEL_PATH holding labels like "Day 1", "Day 2", ...
FID_COLUMN = "FID"   # column in EXCEL_PATH holding the drifter FID to seed each day from

# ============================================================================
# OUTPUTS
# ============================================================================

# All results are written here:
#   <Day N>_actual_track.csv            - observed drifter positions for that day's segment
#   <Day N>_oceandrift_predicted.csv    - OceanDrift model's predicted track
#   <Day N>_plastdrift_predicted.csv    - PlastDrift model's predicted track
#   <Day N>_oceandrift_separation.csv   - actual vs. OceanDrift, merged with separation distance (km)
#   <Day N>_plastdrift_separation.csv   - actual vs. PlastDrift, merged with separation distance (km)
#   combined_track_comparison.png       - one plot with every day's segment overlaid
#   overall_summary.csv                 - mean/max/final separation (km) per day per model
OUTPUT_DIR = r"E:\University\Applied Oceanography\Dissertation\Results\Comparison"

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

SIMULATION_DURATION_DAYS = 1     # length of each day's simulated segment, starting from its scheduled FID
MODEL_TIME_STEP_SECONDS = 900    # internal integration time step for OpenDrift
OUTPUT_EVERY_SECONDS = 1800      # how often OpenDrift writes a position to its output
USE_WAVE_STOKES_DRIFT = True     # whether wave-induced Stokes drift is included in the models

MAX_DAYS_TO_RUN = 5              # only process the first N days in the Excel schedule; set to None to run all

TERMINAL_VELOCITY_M_S = 0.01     # PlastDrift only: vertical rise/sink rate of the simulated particle

# Open-Meteo historical forecast settings, used to build the wind input for each segment
OPENMETEO_MODEL = "italia_meteo_arpae_icon_2i"
WIND_GRID_MARGIN_DEG = 0.02       # padding (degrees) added around the drifter track when building the wind grid

# ============================================================================
# LOAD DRIFTER DATA (shared across every day and both models)
# ============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# Clean up: drop rows with no timestamp, sort chronologically, remove duplicate timestamps
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

# Shared Open-Meteo API session with local caching and automatic retries,
# reused for every day's wind request so repeated calls aren't re-fetched
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)


def build_wind_reader(start_lat, start_lon, pd_start_time, end_time_utc, actual_track_df):
    """
    Fetch historical Open-Meteo wind data covering one day's simulation window
    and wrap it into an OpenDrift-compatible NetCDF reader.

    Inputs:
        start_lat, start_lon   - seed position for the Open-Meteo query
        pd_start_time          - segment start time (used as the forecast start date)
        end_time_utc           - segment end time (used as the forecast end date)
        actual_track_df        - observed drifter points for this segment, used only
                                  to size the small lat/lon grid the wind is placed on

    Output:
        An OpenDrift reader_netCDF_CF_generic.Reader wrapping a 3x3 grid of
        hourly eastward/northward wind components (u10, v10).
    """
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

    # Convert wind speed/direction into eastward (u) and northward (v) components
    direction_rad = np.radians(hourly_wind_direction_10m)
    u10_1d = -hourly_wind_speed_10m * np.sin(direction_rad)
    v10_1d = -hourly_wind_speed_10m * np.cos(direction_rad)

    # Build a small 3x3 grid around the actual track so OpenDrift has spatial coverage,
    # even though the wind values themselves are uniform across the grid
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

    print(f"    Open-Meteo wind data retrieved: {len(hourly_time)} hourly steps "
          f"({hourly_time[0]} to {hourly_time[-1]})")

    return reader_netCDF_CF_generic.Reader(wind_dataset, name="Open-Meteo historical forecast wind")


def run_model(model_class, model_kwargs, wind_reader, start_lon, start_lat,
              dynamic_start_time_py, simulation_duration_delta):
    """
    Run one OpenDrift model (OceanDrift or PlastDrift) for one day's segment.

    Inputs:
        model_class             - OceanDrift or PlastDrift class
        model_kwargs             - extra keyword args passed to seed_elements
                                    (e.g. terminal_velocity for PlastDrift)
        wind_reader               - wind reader built by build_wind_reader()
        start_lon, start_lat      - seed position (drifter's position at segment start)
        dynamic_start_time_py     - seed time as a native Python datetime
        simulation_duration_delta - pd.Timedelta for how long to run the simulation

    Output:
        DataFrame with columns [time_utc, pred_lat, pred_lon], one row per
        model output step, sorted chronologically.
    """
    # Fresh wave/current readers on every call, since reader objects carry internal state
    wave_reader = reader_netCDF_CF_generic.Reader(WAVE_FILE, name="Copernicus waves")
    current_reader = reader_netCDF_CF_generic.Reader(CURRENTS_FILE, name="Copernicus surface currents")

    model = model_class(loglevel=20)
    model.add_reader([wind_reader, wave_reader, current_reader])
    model.set_config("drift:stokes_drift", USE_WAVE_STOKES_DRIFT)
    model.set_config("drift:use_tabularised_stokes_drift", False)

    model.seed_elements(
        lon=start_lon,
        lat=start_lat,
        number=1,
        radius=0,
        time=dynamic_start_time_py,
        z=0,
        **model_kwargs
    )

    model.run(
        duration=simulation_duration_delta,
        time_step=MODEL_TIME_STEP_SECONDS,
        time_step_output=OUTPUT_EVERY_SECONDS,
    )

    predicted_times = pd.to_datetime(model.result.time.values)
    predicted_lons = model.result.lon.values[0, :]
    predicted_lats = model.result.lat.values[0, :]

    return pd.DataFrame({
        "time_utc": predicted_times,
        "pred_lat": predicted_lats,
        "pred_lon": predicted_lons
    }).sort_values("time_utc").reset_index(drop=True)


def compute_separation(actual_track_df, predicted_df):
    """
    Match each actual drifter observation to its nearest model prediction
    (within 30 minutes) and compute the great-circle distance between them.

    Inputs:
        actual_track_df - observed drifter points, needs columns [time_utc, lat, lon]
        predicted_df     - model output, needs columns [time_utc, pred_lat, pred_lon]

    Output:
        Merged DataFrame with an added 'separation_km' column: the haversine
        distance (km) between each actual point and its matched prediction.
        Rows with no prediction within 30 minutes are dropped.
    """
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

    # Haversine distance between actual and predicted positions
    lat1, lon1, lat2, lon2 = map(np.radians,
                                 [merged_df["lat"], merged_df["lon"], merged_df["pred_lat"], merged_df["pred_lon"]])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    haversine_array = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    merged_df["separation_km"] = (2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(haversine_array))) / 1000.0
    return merged_df


# ============================================================================
# MAIN LOOP - run both models for every day in the Excel schedule
# ============================================================================

schedule_df = pd.read_excel(EXCEL_PATH)
schedule_df["_day_num"] = schedule_df[DAY_COLUMN].astype(str).str.extract(r"(\d+)").astype(int)
schedule_df = schedule_df.sort_values("_day_num")

if MAX_DAYS_TO_RUN is not None:
    schedule_df = schedule_df.head(MAX_DAYS_TO_RUN)
    print(f"Limiting run to the first {MAX_DAYS_TO_RUN} day(s) in the schedule.")

# Accumulators for the single combined plot at the end. A NaN is inserted between
# each day's points so consecutive days aren't visually joined by a straight line,
# while each model still only gets one legend entry.
combined_actual_lon, combined_actual_lat = [], []
combined_ocean_lon, combined_ocean_lat = [], []
combined_plast_lon, combined_plast_lat = [], []
day_boundary_lon, day_boundary_lat = [], []   # start/end point of each day's actual track, for markers

overall_summary_rows = []
overall_start_time = None
overall_end_time = None

for _, row in schedule_df.iterrows():
    day_label = f"Day {int(row['_day_num'])}"   # normalised to "Day 1", "Day 2", ... regardless of source format
    fid = row[FID_COLUMN]

    print(f"\n{'=' * 70}")
    print(f"RUNNING {day_label}  (FID {fid})")
    print(f"{'=' * 70}")

    try:
        # Find the drifter observation that seeds this day's simulation
        start_rows = drifter_df.loc[drifter_df["FID"] == fid]
        if start_rows.empty:
            raise ValueError(f"FID {fid} not found in the processed drifter data.")
        first_point = start_rows.iloc[0]

        start_lon = first_point["lon"]
        start_lat = first_point["lat"]
        pd_start_time = first_point["time_utc"]
        dynamic_start_time_py = pd_start_time.to_pydatetime()

        # Segment end time is capped at the last available drifter observation
        target_end_time = pd_start_time + pd.Timedelta(days=SIMULATION_DURATION_DAYS)
        available_end_time = drifter_df["time_utc"].max()
        end_time_utc = pd.Timestamp(min(target_end_time, available_end_time))

        if end_time_utc <= pd_start_time:
            raise ValueError(f"No drifter data available after {day_label}'s start time ({pd_start_time}).")

        simulation_duration_delta = end_time_utc - pd_start_time

        actual_track_df = drifter_df.loc[
            (drifter_df["time_utc"] >= pd_start_time)
            & (drifter_df["time_utc"] <= end_time_utc)
        ].reset_index(drop=True)

        if actual_track_df.empty:
            raise ValueError(f"No observations fall within {day_label}'s [{pd_start_time}, {end_time_utc}] window.")

        print(f"  Segment window: {pd_start_time} -> {end_time_utc}  "
              f"({len(actual_track_df)} actual points)")

        # Track the overall run span, used later in the combined plot's title
        if overall_start_time is None or pd_start_time < overall_start_time:
            overall_start_time = pd_start_time
        if overall_end_time is None or end_time_utc > overall_end_time:
            overall_end_time = end_time_utc

        # Wind data for this segment (shared input for both models)
        print("  Fetching historical forecast wind data from Open-Meteo...")
        wind_reader = build_wind_reader(start_lat, start_lon, pd_start_time, end_time_utc, actual_track_df)

        # Run both drift models on this segment
        print("  Running OceanDrift...")
        ocean_predicted_df = run_model(
            OceanDrift, {}, wind_reader, start_lon, start_lat,
            dynamic_start_time_py, simulation_duration_delta
        )

        print("  Running PlastDrift...")
        plast_predicted_df = run_model(
            PlastDrift, {"terminal_velocity": TERMINAL_VELOCITY_M_S}, wind_reader, start_lon, start_lat,
            dynamic_start_time_py, simulation_duration_delta
        )

        # Compare each model's prediction against the actual observed track
        ocean_merged_df = compute_separation(actual_track_df, ocean_predicted_df)
        plast_merged_df = compute_separation(actual_track_df, plast_predicted_df)

        # Write per-day CSV outputs (no per-day plots - only the combined plot is produced)
        safe_label = day_label.replace(" ", "_")
        actual_track_df.to_csv(os.path.join(OUTPUT_DIR, f"{safe_label}_actual_track.csv"), index=False)
        ocean_predicted_df.to_csv(os.path.join(OUTPUT_DIR, f"{safe_label}_oceandrift_predicted.csv"), index=False)
        plast_predicted_df.to_csv(os.path.join(OUTPUT_DIR, f"{safe_label}_plastdrift_predicted.csv"), index=False)
        ocean_merged_df.to_csv(os.path.join(OUTPUT_DIR, f"{safe_label}_oceandrift_separation.csv"), index=False)
        plast_merged_df.to_csv(os.path.join(OUTPUT_DIR, f"{safe_label}_plastdrift_separation.csv"), index=False)

        # Append this day's points to the combined-plot accumulators (with a NaN break after)
        combined_actual_lon += list(actual_track_df["lon"]) + [np.nan]
        combined_actual_lat += list(actual_track_df["lat"]) + [np.nan]
        combined_ocean_lon += list(ocean_predicted_df["pred_lon"]) + [np.nan]
        combined_ocean_lat += list(ocean_predicted_df["pred_lat"]) + [np.nan]
        combined_plast_lon += list(plast_predicted_df["pred_lon"]) + [np.nan]
        combined_plast_lat += list(plast_predicted_df["pred_lat"]) + [np.nan]

        # Record this day's actual-track start/end points, for the black-dot markers on the plot
        day_boundary_lon += [actual_track_df["lon"].iloc[0], actual_track_df["lon"].iloc[-1]]
        day_boundary_lat += [actual_track_df["lat"].iloc[0], actual_track_df["lat"].iloc[-1]]

        # Summary stats for this day/model, added to the overall summary table
        overall_summary_rows.append({
            "day": day_label, "fid": fid, "model": "OceanDrift",
            "start_utc": pd_start_time, "end_utc": end_time_utc,
            "mean_separation_km": ocean_merged_df["separation_km"].mean(),
            "max_separation_km": ocean_merged_df["separation_km"].max(),
            "final_separation_km": ocean_merged_df["separation_km"].iloc[-1],
        })
        overall_summary_rows.append({
            "day": day_label, "fid": fid, "model": "PlastDrift",
            "start_utc": pd_start_time, "end_utc": end_time_utc,
            "mean_separation_km": plast_merged_df["separation_km"].mean(),
            "max_separation_km": plast_merged_df["separation_km"].max(),
            "final_separation_km": plast_merged_df["separation_km"].iloc[-1],
        })

        print(f"  OceanDrift  - mean {ocean_merged_df['separation_km'].mean():.3f} km, "
              f"max {ocean_merged_df['separation_km'].max():.3f} km")
        print(f"  PlastDrift  - mean {plast_merged_df['separation_km'].mean():.3f} km, "
              f"max {plast_merged_df['separation_km'].max():.3f} km")

    except Exception as e:
        # A failure on one day (e.g. missing FID, no data in window) doesn't stop the rest of the run
        print(f"  ERROR on {day_label} (FID {fid}): {e}")
        continue

# ============================================================================
# COMBINED TRACK PLOT - every day's segment stitched onto one figure
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 9))
ax.plot(combined_actual_lon, combined_actual_lat, "-o", color="blue", label="Actual drifter track",
        markersize=2.5, linewidth=1.2, zorder=3)
ax.plot(combined_ocean_lon, combined_ocean_lat, "-o", color="red", label="OpenDrift track",
        markersize=2.5, linewidth=1.2, zorder=2)
ax.plot(combined_plast_lon, combined_plast_lat, "-o", color="green", label="OpenDrift PlastDrift track",
        markersize=2.5, linewidth=1.2, zorder=2)
ax.scatter(day_boundary_lon, day_boundary_lat, color="black", s=25, zorder=4, label="Day start/end")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

overall_start_str = overall_start_time.strftime("%Y-%m-%d %H:%M UTC") if overall_start_time is not None else "N/A"
overall_end_str = overall_end_time.strftime("%Y-%m-%d %H:%M UTC") if overall_end_time is not None else "N/A"

ax.set_title(
    f"Drifter Trajectory Validation: Actual vs. OpenDrift vs. OpenDrift PlastDrift\n"
    f"Start: {overall_start_str}   |   End: {overall_end_str}"
)
ax.legend()
ax.grid(True, linestyle="--", alpha=0.4)
ax.set_aspect("equal", adjustable="datalim")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "combined_track_comparison.png"), dpi=200)
plt.close(fig)

# ============================================================================
# OVERALL SUMMARY CSV - mean/max/final separation error per day per model
# ============================================================================

overall_summary_df = pd.DataFrame(overall_summary_rows)
overall_summary_df.to_csv(os.path.join(OUTPUT_DIR, "overall_summary.csv"), index=False)

print(f"\nCombined plot saved to: {os.path.join(OUTPUT_DIR, 'combined_track_comparison.png')}")
print(f"Overall summary saved to: {os.path.join(OUTPUT_DIR, 'overall_summary.csv')}")
print(f"All per-day CSVs saved to: {OUTPUT_DIR}")
print("\nAll days processed.")