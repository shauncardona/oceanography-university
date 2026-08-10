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

# INPUT FILES & DIRECTORIES
DRIFTER_CSV = r"E:\University\Applied Oceanography\Dissertation\Comparison Drifters\Drifters & Copernicus\Drifter Data\Drifter 1.csv"
CURRENTS_FILE = r"E:\University\Applied Oceanography\Dissertation\Comparison Drifters\Drifters & Copernicus\data\MonthCurrentsAnalysis.nc"
WAVE_FILE = r"E:\University\Applied Oceanography\Dissertation\Comparison Drifters\Drifters & Copernicus\data\MonthWaveAnalysis.nc"

# OUTPUT DIRECTORY
OUTPUT_DIR = r"E:\University\Applied Oceanography\Dissertation\Comparison Drifters\Drifters & Copernicus\Output\Drifter 1\Day 2"

# ----------------------------------------------------------------------

# CONFIGURATION PARAMETERS
FID_START = 50            # simulation seed position & start time come from this FID's first observation
SIMULATION_DURATION_DAYS = 3  # simulation automatically ends this many days after the start time
MODEL_TIME_STEP_SECONDS = 900
OUTPUT_EVERY_SECONDS = 1800
USE_WAVE_STOKES_DRIFT = True

# Open-Meteo wind grid configuration
WIND_GRID_MARGIN_DEG = 0.2      # extra safety margin (degrees) added around the actual
                                 # drifter track's own lat/lon bounding box, so the fake
                                 # constant-value wind grid comfortably covers wherever
                                 # the simulated particle drifts too

# ----------------------------------------------------------------------

# LOADING ACTUAL DRIFTER DATA
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

drifter_df = (
    drifter_df.dropna(subset=["time_utc"])
    .sort_values("time_utc")
    .drop_duplicates(subset="time_utc")
    .reset_index(drop=True)
)
drifter_df = drifter_df.rename(columns={"Latitude": "lat", "Longitude": "lon"})

if drifter_df.empty:
    raise ValueError(f"Target drifter CSV resulted in an empty dataset: {DRIFTER_CSV}")

start_rows = drifter_df.loc[drifter_df["FID"] == FID_START]
if start_rows.empty:
    raise ValueError(f"FID_START {FID_START} not found in the processed drifter data.")
first_point = start_rows.iloc[0]

START_LON = first_point["lon"]
START_LAT = first_point["lat"]
pd_start_time = first_point["time_utc"]
DYNAMIC_START_TIME_UTC_PY = pd_start_time.to_pydatetime()

# End time is simply N days after the start time, capped to whatever actual data exists
target_end_time = pd_start_time + pd.Timedelta(days=SIMULATION_DURATION_DAYS)
available_end_time = drifter_df["time_utc"].max()
end_time_utc = min(target_end_time, available_end_time)

if end_time_utc <= pd_start_time:
    raise ValueError(
        f"No drifter data available after the FID_START ({FID_START}) start time "
        f"({pd_start_time}). Check your drifter CSV coverage."
    )

print(f"Drifter data loaded. Observations: {len(drifter_df)}")
print(f"DYNAMIC SEEDING matching FID_START {FID_START}:")
print(f"  Time (UTC): {pd_start_time}")
print(f"  Position:   Lat {START_LAT}, Lon {START_LON}")
print(f"SIMULATION END (start + {SIMULATION_DURATION_DAYS} days, capped to available data):")
print(f"  Time (UTC): {end_time_utc}")
if target_end_time > available_end_time:
    print(f"  NOTE: requested {SIMULATION_DURATION_DAYS}-day end ({target_end_time}) exceeds "
          f"available drifter data ({available_end_time}); simulation end was capped.")

simulation_duration_delta = end_time_utc - pd_start_time
print(f"Simulation duration: {simulation_duration_delta}")

# Actual track used for comparison/plots: ALL rows in the CSV that fall within
# the [start, end] simulation window. (FID here is a unique per-observation ID,
# not a repeating drifter-track ID, so we don't filter by FID again here.)
actual_track_df = drifter_df.loc[
    (drifter_df["time_utc"] >= pd_start_time)
    & (drifter_df["time_utc"] <= end_time_utc)
].reset_index(drop=True)

if actual_track_df.empty:
    raise ValueError(
        f"No FID_START ({FID_START}) observations fall within the "
        f"[{pd_start_time}, {end_time_utc}] window."
    )

print(f"Actual track points in simulation window: {len(actual_track_df)}")
print(actual_track_df[["time_utc", "lat", "lon"]].head())
print(actual_track_df[["time_utc", "lat", "lon"]].tail())

# ----------------------------------------------------------------------

# FETCHING WIND DATA FROM OPEN-METEO FORECAST API

print("\nFetching forecast wind data from Open-Meteo...")

cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

url = "https://api.open-meteo.com/v1/forecast"

openmeteo_params = {
    "latitude": START_LAT,
    "longitude": START_LON,
    "hourly": ["wind_speed_10m", "wind_direction_10m"],
    "models": "dwd_icon_seamless",
    "start_date": pd_start_time.strftime("%Y-%m-%d"),
    "end_date": end_time_utc.strftime("%Y-%m-%d"),
    "bounding_box": "33.434,12.957,37.11,18.835",
}

responses = openmeteo.weather_api(url, params=openmeteo_params)

# bounding_box can return multiple grid points across the region - pick the one
# closest to the drifter's start position
def _dist_to_start(r):
    return (r.Latitude() - START_LAT) ** 2 + (r.Longitude() - START_LON) ** 2

response = min(responses, key=_dist_to_start)
print(f"Selected Open-Meteo grid point: {response.Latitude()}N, {response.Longitude()}E "
      f"(nearest of {len(responses)} bounding-box points to drifter start)")

hourly = response.Hourly()
hourly_wind_speed_10m = hourly.Variables(0).ValuesAsNumpy()
hourly_wind_direction_10m = hourly.Variables(1).ValuesAsNumpy()

hourly_time_full = pd.date_range(
    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
    freq=pd.Timedelta(seconds=hourly.Interval()),
    inclusive="left",
).tz_localize(None)

hourly_wind_df = pd.DataFrame({
    "time_utc": hourly_time_full,
    "wind_speed_10m": hourly_wind_speed_10m,
    "wind_direction_10m": hourly_wind_direction_10m,
})

# Trim down to just the simulation window (the API returns a wider past/forecast range)
hourly_wind_df = hourly_wind_df.loc[
    (hourly_wind_df["time_utc"] >= pd_start_time) & (hourly_wind_df["time_utc"] <= end_time_utc)
].reset_index(drop=True)

if hourly_wind_df.empty:
    raise ValueError(
        "Open-Meteo returned no hourly wind rows inside the simulation window "
        f"[{pd_start_time}, {end_time_utc}]. Check past_days_needed/forecast_days_needed."
    )

hourly_time = hourly_wind_df["time_utc"]
hourly_wind_speed_10m = hourly_wind_df["wind_speed_10m"].to_numpy()
hourly_wind_direction_10m = hourly_wind_df["wind_direction_10m"].to_numpy()

# Convert meteorological wind speed/direction (direction = FROM) into eastward/northward components
direction_rad = np.radians(hourly_wind_direction_10m)
u10_1d = -hourly_wind_speed_10m * np.sin(direction_rad)
v10_1d = -hourly_wind_speed_10m * np.cos(direction_rad)

# Build a small constant-value spatial grid so the CF-generic reader treats this as
# gridded data. Sized from the ACTUAL drifter track's own lat/lon bounding box (plus a
# safety margin) rather than an arbitrary offset from the start point, so the domain
# comfortably covers wherever the drifter (and simulated particle) actually goes.
lat_min = actual_track_df["lat"].min() - WIND_GRID_MARGIN_DEG
lat_max = actual_track_df["lat"].max() + WIND_GRID_MARGIN_DEG
lon_min = actual_track_df["lon"].min() - WIND_GRID_MARGIN_DEG
lon_max = actual_track_df["lon"].max() + WIND_GRID_MARGIN_DEG

grid_lats = np.array([lat_min, (lat_min + lat_max) / 2.0, lat_max])
grid_lons = np.array([lon_min, (lon_min + lon_max) / 2.0, lon_max])

print(f"Wind grid domain (from actual track bounding box + {WIND_GRID_MARGIN_DEG}° margin): "
      f"{lat_min:.4f}-{lat_max:.4f}N, {lon_min:.4f}-{lon_max:.4f}E")

u10_3d = np.broadcast_to(u10_1d[:, None, None], (len(hourly_time), 3, 3)).copy()
v10_3d = np.broadcast_to(v10_1d[:, None, None], (len(hourly_time), 3, 3)).copy()

wind_dataset = xr.Dataset(
    data_vars={
        "u10": (["time", "latitude", "longitude"], u10_3d),
        "v10": (["time", "latitude", "longitude"], v10_3d),
    },
    coords={
        "time": hourly_time.to_numpy(),
        "latitude": grid_lats,
        "longitude": grid_lons,
    },
)
wind_dataset["u10"].attrs["standard_name"] = "eastward_wind"
wind_dataset["v10"].attrs["standard_name"] = "northward_wind"

print(f"Open-Meteo wind data retrieved: {len(hourly_time)} hourly steps "
      f"({hourly_time.iloc[0]} to {hourly_time.iloc[-1]})")

# ----------------------------------------------------------------------

# READING ENVIRONMENT FORECAST FILES

wind_reader = reader_netCDF_CF_generic.Reader(wind_dataset, name="Open-Meteo forecast wind")
wave_reader = reader_netCDF_CF_generic.Reader(WAVE_FILE, name="Copernicus waves")
current_reader = reader_netCDF_CF_generic.Reader(CURRENTS_FILE, name="Copernicus surface currents")

# ----------------------------------------------------------------------

# RUNNING OPENDRIFT

model = OceanDrift(loglevel=20)
model.add_reader([wind_reader, wave_reader, current_reader])

model.set_config("drift:stokes_drift", USE_WAVE_STOKES_DRIFT)
model.set_config("drift:use_tabularised_stokes_drift", False)

model.seed_elements(
    lon=START_LON,
    lat=START_LAT,
    number=1,
    radius=0,
    time=DYNAMIC_START_TIME_UTC_PY,
    z=0
)

model.run(
    duration=simulation_duration_delta,
    time_step=MODEL_TIME_STEP_SECONDS,
    time_step_output=OUTPUT_EVERY_SECONDS,
)

# ----------------------------------------------------------------------

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

# ----------------------------------------------------------------------

# SAVING OUTPUT DATA & PLOTS

actual_track_df.to_csv(os.path.join(OUTPUT_DIR, "drifter_actual_track.csv"), index=False)
predicted_df.to_csv(os.path.join(OUTPUT_DIR, "predicted_track.csv"), index=False)
merged_df.to_csv(os.path.join(OUTPUT_DIR, "separation_distances.csv"), index=False)

fig, ax = plt.subplots(figsize=(9, 8))
ax.plot(actual_track_df["lon"], actual_track_df["lat"], "-o", color="blue", label="Actual drifter track",
        markersize=3, linewidth=1.5, zorder=3)
ax.plot(predicted_df["pred_lon"], predicted_df["pred_lat"], "-o", color="red", label="OpenDrift track",
        markersize=3, linewidth=1.5, zorder=2)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Drifter Trajectory Validation: Actual vs. OpenDrift")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.4)
ax.set_aspect("equal", adjustable="datalim")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "track_comparison.png"), dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(merged_df["time_utc"], merged_df["separation_km"], "-o", color="black", markersize=3)
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Separation distance (km)")
ax.set_title("Distance Between Observed and OpenDrift Positions")
ax.grid(True, linestyle="--", alpha=0.4)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "separation_distance.png"), dpi=200)
plt.close(fig)

print(f"\nMean separation error:  {merged_df['separation_km'].mean():.3f} km")
print(f"Max separation error:   {merged_df['separation_km'].max():.3f} km")
print(f"Final separation error: {merged_df['separation_km'].iloc[-1]:.3f} km")
print(f"\nAll outputs successfully saved to: {OUTPUT_DIR}")