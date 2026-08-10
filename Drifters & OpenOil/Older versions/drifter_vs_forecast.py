# LOADING LIBRARIES

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from opendrift.models.oceandrift import OceanDrift
from opendrift.readers import reader_netCDF_CF_generic

# Constants
EARTH_RADIUS_M = 6371000.0

# ----------------------------------------------------------------------

# INPUT FILES & DIRECTORIES
DRIFTER_CSV = r"E:\University\Applied Oceanography\Dissertation\Comparison Drifters\Drifters & Copernicus\Drifter Data\SST_DATA_11.csv"
CURRENTS_FILE = r"E:\University\Applied Oceanography\Dissertation\Comparison Drifters\Drifters & Copernicus\data\Week 1\Currents.nc"
WAVE_FILE = r"E:\University\Applied Oceanography\Dissertation\Comparison Drifters\Drifters & Copernicus\data\Week 1\waves.nc"

# Wind directory and targeted consolidated file
WIND_DIR = r"E:\University\Applied Oceanography\Dissertation\Comparison Drifters\Drifters & Copernicus\data\Wind"
WIND_FILE = os.path.join(WIND_DIR, "Wind.grib2")

# OUTPUT DIRECTORY
OUTPUT_DIR = r"E:\University\Applied Oceanography\Dissertation\Comparison Drifters\Drifters & Copernicus\Output"

# ----------------------------------------------------------------------

# CONFIGURATION PARAMETERS
TARGET_FID = 50
MODEL_TIME_STEP_SECONDS = 900
OUTPUT_EVERY_SECONDS = 1800
USE_WAVE_STOKES_DRIFT = True

# ----------------------------------------------------------------------

# AUTOMATIC WIND DATA DOWNLOAD BLOCK (July 6th to July 13th, 2026)

os.makedirs(WIND_DIR, exist_ok=True)

if not os.path.exists(WIND_FILE):
    print("\nTarget wind file not found locally. Initiating automated AWS download pool...")
    try:
        from ecmwf_opendata import Client

        client = Client(source="aws")

        print("Streaming 10m Wind U/V vectors from July 6th to July 13th, 2026...")
        client.retrieve(
            {
                "date": "20260706/to/20260713",
                "time": "00:00:00",
                "step": [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48],
                "stream": "oper",
                "type": "fc",
                "levtype": "sfc",
                "param": ["10u", "10v"],
            },
            WIND_FILE
        )
        print(f"Success! Integrated wind dataset saved to: {WIND_FILE}\n")
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "\n\nMissing dependency! Please run this command in your terminal first:\n"
            "C:\\Users\\Shaun-PC\\miniconda3\\envs\\comparison-drifters\\python.exe -m pip install ecmwf-opendata\n"
        )
    except Exception as e:
        raise RuntimeError(f"Failed streaming environmental wind vectors via AWS: {e}")
else:
    print(f"\nVerified local wind dataset exists: {WIND_FILE}. Skipping download step.")


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

if TARGET_FID is None:
    seed_fid = drifter_df["FID"].min()
else:
    seed_fid = TARGET_FID

seed_rows = drifter_df.loc[drifter_df["FID"] == seed_fid]
if seed_rows.empty:
    raise ValueError(f"FID {seed_fid} not found in the processed drifter data.")
first_point = seed_rows.iloc[0]

START_LON = first_point["lon"]
START_LAT = first_point["lat"]
pd_start_time = first_point["time_utc"]
DYNAMIC_START_TIME_UTC_PY = pd_start_time.to_pydatetime()

print(f"Drifter data loaded. Observations: {len(drifter_df)}")
print(f"DYNAMIC SEEDING matching FID {seed_fid}:")
print(f"  Time (UTC): {pd_start_time}")
print(f"  Position:   Lat {START_LAT}, Lon {START_LON}")

end_time_utc = drifter_df["time_utc"].max()
simulation_duration_delta = end_time_utc - pd_start_time

# ----------------------------------------------------------------------

# READING ENVIRONMENT FORECAST FILES

wind_dataset = xr.open_dataset(WIND_FILE, engine="cfgrib", backend_kwargs={"indexpath": ""})
if "step" in wind_dataset.dims and "valid_time" in wind_dataset.coords:
    wind_dataset = wind_dataset.rename({"time": "reference_time"})
    wind_dataset = wind_dataset.swap_dims({"step": "valid_time"})
    wind_dataset = wind_dataset.rename({"valid_time": "time"})

if "u10" in wind_dataset.variables:
    wind_dataset["u10"].attrs["standard_name"] = "eastward_wind"
if "v10" in wind_dataset.variables:
    wind_dataset["v10"].attrs["standard_name"] = "northward_wind"

wind_reader = reader_netCDF_CF_generic.Reader(wind_dataset, name="ECMWF operational wind")
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

df_actual_calc = drifter_df.copy()
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

drifter_df.to_csv(os.path.join(OUTPUT_DIR, "drifter_actual_track.csv"), index=False)
predicted_df.to_csv(os.path.join(OUTPUT_DIR, "predicted_track.csv"), index=False)
merged_df.to_csv(os.path.join(OUTPUT_DIR, "separation_distances.csv"), index=False)

fig, ax = plt.subplots(figsize=(9, 8))
ax.plot(drifter_df["lon"], drifter_df["lat"], "-o", color="blue", label="Actual drifter track", markersize=3,
        linewidth=1.5)
ax.plot(predicted_df["pred_lon"], predicted_df["pred_lat"], "-o", color="red", label="OpenDrift track", markersize=3,
        linewidth=1.5)
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