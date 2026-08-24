# LOADING LIBRARIES

import os
import csv
import math
import random
from datetime import datetime, timedelta

import xarray as xr
from opendrift.models.openoil import OpenOil
from opendrift.readers import reader_netCDF_CF_generic
from global_land_mask import globe

# ---------------------------------------------------------
# INPUT FILES

WIND_FILE = r"E:\University\Applied Oceanography\Dissertation\Data\Oil Spill Code Data\Wind1.grib"
WAVE_FILE = r"E:\University\Applied Oceanography\Dissertation\Data\Oil Spill Code Data\Waves1.nc"
CURRENTS_FILE = r"E:\University\Applied Oceanography\Dissertation\Data\Oil Spill Code Data\Currents1.nc"

# CSV listing the vessels/spill scenarios to randomly draw from. See the
# accompanying vessels.csv for the expected columns.
VESSELS_FILE = r"E:\University\Applied Oceanography\Dissertation\Data\Oil Spill Code Data\vessels.csv"

# CSV listing candidate spill coordinates (e.g. exported from Google Maps /
# a GPS tool). Must contain "Latitude", "Longitude" and "Label" columns (as
# in coordinates_2026-08-24_0118.csv). Rows are grouped by "Label" (e.g.
# "Malta North", "Malta South"); the min/max of each group's points defines
# that sector's bounding box. Each run's spill start point is drawn at
# random from *inside* the box of the sector it belongs to.
COORDINATES_FILE = r"E:\University\Applied Oceanography\Dissertation\Data\Oil Spill Code Data\coordinates_2026-08-24_0118.csv"

# Which sector(s) to actually run. Each entry must match a "Label" value
# found in COORDINATES_FILE exactly (case-insensitive, whitespace trimmed).
# The full set of runs (all seasons x RUNS_PER_SEASON) is repeated once per
# sector listed here.
SECTORS_TO_RUN = ["Malta North","Malta South"]

# ---------------------------------------------------------
# OUTPUT FOLDERS
# Each run writes its own CSV and PNG into these folders, named after the
# season, run number, spill date, start position and vessel used.

OUTPUT_CSV_DIR = r"E:\University\Applied Oceanography\Dissertation\Results\Malta Oil Spill\Spillcsv"
OUTPUT_IMAGE_DIR = r"E:\University\Applied Oceanography\Dissertation\Results\Malta Oil Spill\imagetrajectory"

# ---------------------------------------------------------
# SIMULATION PARAMETERS

# NOTE: START_LON / START_LAT are no longer fixed constants. Each run now
# draws a random start position from inside the bounding box computed from
# COORDINATES_FILE (see load_spill_box() and main()).

# year to draw random spill start dates from
# IMPORTANT: your WIND_FILE / WAVE_FILE / CURRENTS_FILE must cover the
# whole of this year (all 4 seasons), otherwise runs for months outside
# the data range will fail.
YEAR = 2025

# length of each simulation
SIM_DURATION_DAYS = 4

# how many random runs to do per season (4 seasons x 5 = 20 total runs)
RUNS_PER_SEASON = 15

# meteorological seasons (Mediterranean / Northern hemisphere convention)
SEASONS = {
    "winter": [12, 1, 2],
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "autumn": [9, 10, 11],
}

# set an integer to make the random dates/vessels/positions reproducible
# run to run, or leave as None for a different draw every time you execute
# the script
RANDOM_SEED = None

# duration of spill (0 for an instantaneous release)
SPILL_DURATION_HOURS = 0

# number of particles
NUMBER_OF_PARTICLES = 500

# how widely particles are initially spread around the point (in meters)
SEED_RADIUS_METERS = 0

# model and output time steps
MODEL_TIME_STEP_SECONDS = 900
OUTPUT_EVERY_SECONDS = 3600

# use Stokes drift (needs VSDX and VSDY parameters in waves file)
USE_WAVE_STOKES_DRIFT = True

# use wave mixing (needs VHM0, VTM02, VTPK in waves file)
USE_WAVE_MIXING = True

# use weathering
USE_OIL_WEATHERING = True

# plot extent
PLOT_LON_MIN = 12.96
PLOT_LON_MAX = 15.84
PLOT_LAT_MIN = 35.434
PLOT_LAT_MAX = 37.110

# ---------------------------------------------------------
# SPILL-BOX LOADING


def load_spill_boxes(path):
    """
    Reads a CSV of candidate coordinates (must contain "Label", "Latitude"
    and "Longitude" columns, as exported from a GPS/coordinate tool).
    Rows are grouped by "Label" (sector name), and for each sector the
    min/max lon/lat of its points is computed.

    Returns a dict: {sector_name: {"lon_min", "lon_max", "lat_min", "lat_max"}}
    Sector names are matched case-insensitively / whitespace-trimmed
    elsewhere (see get_box_by_name), but are stored here exactly as they
    appear in the CSV.
    """
    points_by_sector = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lat = float(row["Latitude"])
                lon = float(row["Longitude"])
            except (KeyError, ValueError, TypeError):
                continue
            label = (row.get("Label") or "").strip()
            if not label:
                continue
            points_by_sector.setdefault(label, {"lats": [], "lons": []})
            points_by_sector[label]["lats"].append(lat)
            points_by_sector[label]["lons"].append(lon)

    if not points_by_sector:
        raise ValueError(f"No valid Label/Latitude/Longitude rows found in {path}")

    boxes = {}
    for label, pts in points_by_sector.items():
        boxes[label] = {
            "lon_min": min(pts["lons"]),
            "lon_max": max(pts["lons"]),
            "lat_min": min(pts["lats"]),
            "lat_max": max(pts["lats"]),
        }
    return boxes


def get_box_by_name(boxes, sector_name):
    """
    Looks up a sector's bounding box by name, matching case-insensitively
    and ignoring leading/trailing whitespace. Raises a clear error listing
    the available sector names if there's no match.
    """
    target = sector_name.strip().lower()
    for label, box in boxes.items():
        if label.strip().lower() == target:
            return box
    available = ", ".join(sorted(boxes.keys()))
    raise ValueError(
        f"Sector {sector_name!r} not found in coordinates file. "
        f"Available sectors: {available}"
    )


def random_point_in_box(box):
    """Returns a random (lon, lat) uniformly drawn from inside the box."""
    lon = random.uniform(box["lon_min"], box["lon_max"])
    lat = random.uniform(box["lat_min"], box["lat_max"])
    return lon, lat


def random_ocean_point_in_box(box, max_attempts=500):
    """
    Draws a random (lon, lat) from inside the box, rejecting any point that
    falls on land (checked with the global_land_mask package). Rectangular
    boxes built from a few coastal reference points will often include
    slivers of coastline or whole islands (e.g. Malta/Gozo sit inside the
    Malta North/South boxes), so a plain uniform draw can occasionally
    land on dry ground - which then makes OpenDrift fail immediately with
    "No ocean pixels nearby, cannot move elements." when SEED_RADIUS_METERS
    is 0.

    Raises RuntimeError if no ocean point is found within max_attempts,
    which usually means the box is mostly/entirely land.
    """
    for _ in range(max_attempts):
        lon, lat = random_point_in_box(box)
        if not globe.is_land(lat, lon):
            return lon, lat
    raise RuntimeError(
        f"Could not find an ocean point in box {box} after {max_attempts} "
        "attempts - check that this sector's bounding box actually covers "
        "open water."
    )


# ---------------------------------------------------------
# VESSEL / OIL LOADING


def load_vessels(path):
    """
    Reads a CSV file with columns:
        vessel_name, oil_type, oil_volume_m3, oil_density_kg_per_m3

    oil_type must match a name in the ADIOS oil database
    (https://adios.orr.noaa.gov/oils), same as OIL_TYPE in the original script.

    Returns a list of dicts, one per vessel.
    """
    vessels = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vessels.append(
                {
                    "vessel_name": row["vessel_name"],
                    "oil_type": row["oil_type"],
                    "oil_volume_m3": float(row["oil_volume_m3"]),
                    "oil_density_kg_per_m3": float(row["oil_density_kg_per_m3"]),
                }
            )
    if not vessels:
        raise ValueError(f"No vessels found in {path}")
    return vessels


def validate_oil_types(vessels):
    """
    Checks every vessel's oil_type against OpenOil's allowed ADIOS oil names
    *before* any simulation runs, so a typo fails immediately with a clear
    message instead of mid-batch after time has already been spent.
    """
    checker = OpenOil(loglevel=50)  # loglevel=50 keeps this quiet
    problems = []
    for vessel in vessels:
        try:
            checker.set_config("seed:oil_type", vessel["oil_type"])
        except ValueError:
            problems.append(f"{vessel['vessel_name']!r} -> {vessel['oil_type']!r}")
    if problems:
        raise ValueError(
            "Invalid oil_type found in vessels.csv for:\n  "
            + "\n  ".join(problems)
            + "\nCheck exact spelling against the ADIOS database "
              "(https://adios.orr.noaa.gov/oils)."
        )


# ---------------------------------------------------------
# RANDOM DATE HELPER


def random_date_in_season(season_months, year):
    """Returns a random datetime within the given list of months of `year`."""
    month = random.choice(season_months)
    if month == 12:
        days_in_month = 31
    else:
        next_month = datetime(year, month % 12 + 1, 1)
        days_in_month = (next_month - datetime(year, month, 1)).days
    day = random.randint(1, days_in_month)
    hour = random.randint(0, 23)
    return datetime(year, month, day, hour)


# ---------------------------------------------------------
# READERS (built once and reused across all runs)


def build_readers():
    wind_dataset = xr.open_dataset(
        WIND_FILE,
        engine="cfgrib",
        backend_kwargs={"indexpath": ""},
    )
    wind_dataset["u10"].attrs["standard_name"] = "eastward_wind"
    wind_dataset["v10"].attrs["standard_name"] = "northward_wind"
    wind_reader = reader_netCDF_CF_generic.Reader(wind_dataset, name="ERA5 wind")

    wave_reader = reader_netCDF_CF_generic.Reader(WAVE_FILE, name="Copernicus waves")

    current_reader = reader_netCDF_CF_generic.Reader(
        CURRENTS_FILE,
        name="Copernicus surface currents",
    )

    return wind_reader, wave_reader, current_reader


# ---------------------------------------------------------
# SINGLE SIMULATION RUN


def run_simulation(readers, start_time, end_time, start_lon, start_lat, vessel, csv_path, png_path):
    wind_reader, wave_reader, current_reader = readers

    model = OpenOil(loglevel=20)
    model.add_reader([wind_reader, wave_reader, current_reader])

    model.set_config("drift:stokes_drift", USE_WAVE_STOKES_DRIFT)
    model.set_config("drift:use_tabularised_stokes_drift", False)

    model.set_config("drift:vertical_mixing", USE_WAVE_MIXING)
    model.set_config("processes:dispersion", USE_WAVE_MIXING)
    model.set_config("processes:evaporation", USE_OIL_WEATHERING)
    model.set_config("processes:emulsification", USE_OIL_WEATHERING)

    if SPILL_DURATION_HOURS == 0:
        seed_time = start_time
        m3_per_hour = vessel["oil_volume_m3"]
    else:
        spill_end_time = start_time + timedelta(hours=SPILL_DURATION_HOURS)
        seed_time = [start_time, spill_end_time]
        m3_per_hour = vessel["oil_volume_m3"] / SPILL_DURATION_HOURS

    model.seed_elements(
        lon=start_lon,
        lat=start_lat,
        radius=SEED_RADIUS_METERS,
        number=NUMBER_OF_PARTICLES,
        time=seed_time,
        z=0,
        oil_type=vessel["oil_type"],
        m3_per_hour=m3_per_hour,
    )

    # use vessel oil_type for oil properties, but keep the mass budget equal
    # to oil_volume_m3 * oil_density_kg_per_m3
    model.elements_scheduled.mass_oil = (
        vessel["oil_volume_m3"] * vessel["oil_density_kg_per_m3"] / NUMBER_OF_PARTICLES
    )
    model.elements_scheduled.density = vessel["oil_density_kg_per_m3"]

    model.run(
        duration=end_time - start_time,
        time_step=MODEL_TIME_STEP_SECONDS,
        time_step_output=OUTPUT_EVERY_SECONDS,
    )

    # --- writing CSV ---
    times = model.result.time.values
    lons = model.result.lon.values
    lats = model.result.lat.values
    statuses = model.result.status.values
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["trajectory", "time", "longitude", "latitude", "status"])
        for trajectory_index in range(lons.shape[0]):
            for time_index in range(lons.shape[1]):
                lon = float(lons[trajectory_index, time_index])
                lat = float(lats[trajectory_index, time_index])
                if not (math.isfinite(lon) and math.isfinite(lat)):
                    continue
                writer.writerow(
                    [
                        trajectory_index,
                        str(times[time_index]),
                        lon,
                        lat,
                        int(statuses[trajectory_index, time_index]),
                    ]
                )

    # --- writing plot ---
    model.plot(
        filename=png_path,
        fast=True,
        corners=[PLOT_LON_MIN, PLOT_LON_MAX, PLOT_LAT_MIN, PLOT_LAT_MAX],
    )


# ---------------------------------------------------------
# MAIN: for each sector in SECTORS_TO_RUN, N runs per season, each with a
# random date, a random start position drawn from that sector's bounding
# box, and a random vessel. Total runs = len(SECTORS_TO_RUN) x len(SEASONS)
# x RUNS_PER_SEASON.


def main():
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

    vessels = load_vessels(VESSELS_FILE)
    validate_oil_types(vessels)

    all_boxes = load_spill_boxes(COORDINATES_FILE)
    for sector_name in SECTORS_TO_RUN:
        box = get_box_by_name(all_boxes, sector_name)
        print(
            f"Sector {sector_name!r} box loaded: "
            f"lon [{box['lon_min']:.4f}, {box['lon_max']:.4f}], "
            f"lat [{box['lat_min']:.4f}, {box['lat_max']:.4f}]"
        )

    readers = build_readers()

    log_rows = []

    for sector_name in SECTORS_TO_RUN:
        spill_box = get_box_by_name(all_boxes, sector_name)
        sector_tag = sector_name.strip().replace(" ", "_")

        for season_name, season_months in SEASONS.items():
            for run_number in range(1, RUNS_PER_SEASON + 1):
                start_time = random_date_in_season(season_months, YEAR)
                end_time = start_time + timedelta(days=SIM_DURATION_DAYS)
                vessel = random.choice(vessels)
                start_lon, start_lat = random_ocean_point_in_box(spill_box)

                date_tag = start_time.strftime("%Y%m%d_%H%M")
                pos_tag = f"{start_lat:.3f}N_{start_lon:.3f}E"
                base_name = (
                    f"{sector_tag}_{season_name}_run{run_number}_{date_tag}"
                    f"_{pos_tag}_{vessel['vessel_name']}"
                )

                csv_path = os.path.join(OUTPUT_CSV_DIR, f"{base_name}.csv")
                png_path = os.path.join(OUTPUT_IMAGE_DIR, f"{base_name}.png")

                print(
                    f"Running {base_name}: sector={sector_name}, start={start_time}, "
                    f"position=({start_lon:.4f}, {start_lat:.4f}), "
                    f"vessel={vessel['vessel_name']}, oil={vessel['oil_type']}, "
                    f"volume={vessel['oil_volume_m3']} m3"
                )

                run_simulation(
                    readers, start_time, end_time, start_lon, start_lat, vessel, csv_path, png_path
                )

                log_rows.append(
                    {
                        "sector": sector_name,
                        "season": season_name,
                        "run": run_number,
                        "start_time": start_time,
                        "end_time": end_time,
                        "start_lon": start_lon,
                        "start_lat": start_lat,
                        "vessel": vessel["vessel_name"],
                        "oil_type": vessel["oil_type"],
                        "oil_volume_m3": vessel["oil_volume_m3"],
                        "csv_file": csv_path,
                        "png_file": png_path,
                    }
                )

    # summary of all runs, so you can trace which scenario produced which file
    log_path = os.path.join(OUTPUT_CSV_DIR, "run_summary.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"\nAll runs complete. Summary written to {log_path}")


if __name__ == "__main__":
    main()