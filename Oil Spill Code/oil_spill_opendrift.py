# LOADING LIBRARIES

import os
import csv
import math
import random
import traceback
from datetime import datetime, timedelta

# Force the non-interactive Agg backend BEFORE any other matplotlib-using
# import (opendrift imports matplotlib.pyplot internally). On Windows, the
# default interactive backend allocates real OS-level GDI bitmap handles
# per figure; across thousands of batch runs these can accumulate faster
# than they're released and eventually exhaust the process's GDI handle
# limit, crashing the whole process with "Fail to allocate bitmap" - a
# native crash that a Python try/except cannot catch after the fact. Agg
# renders entirely in memory and avoids this class of crash.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import xarray as xr
from opendrift.models.openoil import OpenOil
from opendrift.readers import reader_netCDF_CF_generic
from global_land_mask import globe

# ---------------------------------------------------------
# INPUT FILES

WIND_FILE = r"E:\University\Applied Oceanography\Dissertation\Data\Oil Spill Code Data\Wind2.grib"
WAVE_FILE = r"E:\University\Applied Oceanography\Dissertation\Data\Oil Spill Code Data\Waves2.nc"
CURRENTS_FILE = r"E:\University\Applied Oceanography\Dissertation\Data\Oil Spill Code Data\Currents2.nc"

# CSV listing candidate spill coordinates (e.g. exported from Google Maps /
# a GPS tool). Must contain "Latitude", "Longitude" and "Label" columns (as
# in coordinates_2026-08-24_0118.csv). Rows are grouped by "Label" (e.g.
# "Malta North", "Malta South"); the min/max of each group's points defines
# that sector's bounding box. Each run's spill start point is drawn at
# random from *inside* the box of the sector it belongs to.
COORDINATES_FILE = r"E:\University\Applied Oceanography\Dissertation\Data\Oil Spill Code Data\coordinates.csv"

# Which sector(s) to actually run. Each entry must match a "Label" value
# found in COORDINATES_FILE exactly (case-insensitive, whitespace trimmed).
# The full set of runs (all seasons x RUNS_PER_SEASON) is repeated once per
# sector listed here.
SECTORS_TO_RUN = ["Malta North"]

# ---------------------------------------------------------
# OIL SPILL SCENARIOS
# Each run randomly picks one of two spill categories, each with its own
# set of valid ADIOS oil_type names and its own realistic total-mass range
# (in tonnes). A crude-oil spill (tanker cargo) is orders of magnitude
# larger than a heavy-fuel-oil spill (a vessel's own bunker fuel), so each
# category gets its own scale rather than sharing one volume range.

OIL_SCENARIOS = {
    "crude_oil": {
        "oil_types": ["GENERIC HEAVY CRUDE"],
        "tonnes_min": 50_000,
        "tonnes_max": 300_000,
    },
    "heavy_fuel_oil": {
        "oil_types": ["GENERIC HEAVY FUEL OIL", "BUNKER C FUEL OIL"],
        "tonnes_min": 1_000,
        "tonnes_max": 5_000,
    },
}

# ---------------------------------------------------------
# OUTPUT FOLDERS
# Each run writes its own CSV and PNG into these folders, named after the
# season, run number, spill date, start position and oil type used.

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

# how many random runs to do per calendar day (one folder is created per
# day, containing all of that day's runs)
RUNS_PER_DAY = 15

# meteorological seasons (Mediterranean / Northern hemisphere convention)
SEASONS = {
    "winter": [12, 1, 2],
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "autumn": [9, 10, 11],
}

# reverse lookup: month number -> season name, built once from SEASONS
MONTH_TO_SEASON = {
    month: season_name for season_name, months in SEASONS.items() for month in months
}

# set an integer to make the random dates/oil types/volumes/positions/
# spill-types reproducible run to run, or leave as None for a different
# draw every time you execute the script
RANDOM_SEED = None

# --- Spill type (drawn at random, per run) ---
# Each run is randomly either:
#   "instantaneous" - a single-moment release; the initial particle patch
#       is spread over a random radius between INSTANTANEOUS_RADIUS_MIN_M
#       and INSTANTANEOUS_RADIUS_MIN_M meters.
#   "continuous" - a steady release lasting CONTINUOUS_DURATION_CHOICES_HOURS
#       hours (1 or 2 days), seeded as a point source (radius 0), matching
#       how a leaking/drifting vessel would be modelled.
SPILL_TYPES = ["instantaneous", "continuous"]

# random radius range (meters) used for instantaneous spills
INSTANTANEOUS_RADIUS_MIN_M = 50
INSTANTANEOUS_RADIUS_MAX_M = 100

# possible continuous-release durations (hours) - chosen at random per run
CONTINUOUS_DURATION_CHOICES_HOURS = [24, 48]

# number of particles
NUMBER_OF_PARTICLES = 500

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
PLOT_LAT_MIN = 35.5
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
# OIL TYPE VALIDATION / DENSITY LOOKUP


def all_oil_types(oil_scenarios):
    """Flattens the oil_types lists across every category into one list."""
    types = []
    for scenario in oil_scenarios.values():
        types.extend(scenario["oil_types"])
    return types


def validate_oil_types(oil_types):
    """
    Checks every configured oil_type against OpenOil's allowed ADIOS oil
    names *before* any simulation runs, so a typo (or an unreachable ADIOS
    lookup) fails immediately with a clear message instead of mid-batch
    after time has already been spent.
    """
    checker = OpenOil(loglevel=50)  # loglevel=50 keeps this quiet
    problems = []
    for oil_type in oil_types:
        try:
            checker.set_config("seed:oil_type", oil_type)
        except ValueError:
            problems.append(oil_type)
    if problems:
        raise ValueError(
            "Invalid oil_type found in OIL_SCENARIOS:\n  "
            + "\n  ".join(repr(p) for p in problems)
            + "\nCheck exact spelling against the ADIOS database "
              "(https://adios.orr.noaa.gov/oils)."
        )


def build_oil_density_cache(oil_types):
    """
    Looks up each oil_type's real ADIOS density once, up front, by seeding
    a single throwaway element with fallback (zeroed) environment values -
    no reader files are needed for this. Returns {oil_type: density_kg_m3}.

    This lets each run convert its target spill mass (tonnes) into an
    approximate volume for seed_elements()'s m3_per_hour argument, using
    the oil's real density rather than a generic guess.
    """
    cache = {}
    for oil_type in oil_types:
        probe = OpenOil(loglevel=50)
        probe.set_config("seed:oil_type", oil_type)
        for var in ("x_sea_water_velocity", "y_sea_water_velocity", "x_wind", "y_wind"):
            probe.set_config(f"environment:fallback:{var}", 0)
        probe.seed_elements(
            lon=0, lat=0, radius=0, number=1, time=datetime(2000, 1, 1), z=0
        )
        cache[oil_type] = float(probe.elements_scheduled.density)
    return cache


# ---------------------------------------------------------
# DATE HELPER


def all_dates_in_year(year):
    """Yields one datetime (at midnight) per calendar day of `year`."""
    current = datetime(year, 1, 1)
    end = datetime(year + 1, 1, 1)
    while current < end:
        yield current
        current += timedelta(days=1)


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


def run_simulation(
    readers,
    start_time,
    end_time,
    start_lon,
    start_lat,
    oil_type,
    oil_mass_tonnes,
    oil_volume_m3,
    spill_duration_hours,
    seed_radius_meters,
    sector_name,
    spill_type,
    csv_path,
    png_path,
):
    wind_reader, wave_reader, current_reader = readers

    model = OpenOil(loglevel=20)
    model.add_reader([wind_reader, wave_reader, current_reader])

    model.set_config("drift:stokes_drift", USE_WAVE_STOKES_DRIFT)
    model.set_config("drift:use_tabularised_stokes_drift", False)

    model.set_config("drift:vertical_mixing", USE_WAVE_MIXING)
    model.set_config("processes:dispersion", USE_WAVE_MIXING)
    model.set_config("processes:evaporation", USE_OIL_WEATHERING)
    model.set_config("processes:emulsification", USE_OIL_WEATHERING)

    if spill_duration_hours == 0:
        seed_time = start_time
        m3_per_hour = oil_volume_m3
    else:
        spill_end_time = start_time + timedelta(hours=spill_duration_hours)
        seed_time = [start_time, spill_end_time]
        m3_per_hour = oil_volume_m3 / spill_duration_hours

    model.seed_elements(
        lon=start_lon,
        lat=start_lat,
        radius=seed_radius_meters,
        number=NUMBER_OF_PARTICLES,
        time=seed_time,
        z=0,
        oil_type=oil_type,
        m3_per_hour=m3_per_hour,
    )

    # oil_volume_m3 was only an approximation (target mass / cached density)
    # used to give seed_elements() a realistic flow rate / initial slick
    # size. The mass budget itself is set directly and exactly from the
    # target tonnage, spread evenly across all particles - no need to
    # re-derive it from density here.
    oil_mass_kg = oil_mass_tonnes * 1000.0
    model.elements_scheduled.mass_oil = oil_mass_kg / NUMBER_OF_PARTICLES

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
    # Build a status breakdown (e.g. "Active: 45%, Stranded: 55%") from the
    # final recorded status of each particle, so the image is meaningful
    # without needing to cross-reference run_summary.csv. Some particles may
    # have no valid status at the final timestep (e.g. not yet seeded, for
    # a continuous release whose seeding window extends close to the end
    # of the simulation) - these show up as NaN and are reported separately
    # as "No data" rather than breaking the count.
    final_statuses = statuses[:, -1]
    total_particles = final_statuses.shape[0]
    status_names = getattr(model, "status_categories", None)

    def status_label(code):
        if status_names is not None and 0 <= code < len(status_names):
            return status_names[code].replace("_", " ").capitalize()
        return f"Status {code}"

    valid_mask = ~np.isnan(final_statuses)
    valid_statuses = final_statuses[valid_mask]
    nan_count = total_particles - valid_statuses.shape[0]

    unique_codes, counts = np.unique(valid_statuses, return_counts=True)
    breakdown_pairs = [(status_label(int(code)), count) for code, count in zip(unique_codes, counts)]
    if nan_count > 0:
        breakdown_pairs.append(("No data", nan_count))

    breakdown_parts = [
        f"{label} {count / total_particles:.0%}"
        for label, count in sorted(breakdown_pairs, key=lambda x: -x[1])
    ]
    status_breakdown = ", ".join(breakdown_parts)

    # Describe the spill scenario itself
    if spill_type == "instantaneous":
        spill_desc = f"Instantaneous release, seed radius {seed_radius_meters:.0f} m"
    else:
        spill_desc = f"Continuous release over {spill_duration_hours:.0f} h"

    title = (
        f"{sector_name} | {start_time.strftime('%Y-%m-%d %H:%M')} UTC | "
        f"{oil_type} ({oil_mass_tonnes:,.0f} tonnes)\n"
        f"{spill_desc} | Start: {start_lat:.3f}N, {start_lon:.3f}E | "
        f"Simulated {(end_time - start_time).days} days\n"
        f"Final status: {status_breakdown}"
    )

    model.plot(
        filename=png_path,
        fast=True,
        corners=[PLOT_LON_MIN, PLOT_LON_MAX, PLOT_LAT_MIN, PLOT_LAT_MAX],
        title=title,
        legend=True,
    )

    # Explicitly release every figure this run created. Over thousands of
    # runs, unclosed matplotlib figures accumulate in memory (and, on the
    # default Windows backend, as GDI bitmap handles) until the process
    # crashes - closing here prevents that build-up regardless of exactly
    # how many figures model.plot() itself left open.
    plt.close("all")


# ---------------------------------------------------------
# MAIN: for each sector in SECTORS_TO_RUN, loop over every calendar day of
# YEAR. For each day, create one folder (named "<season>_<YYYY-MM-DD>")
# and run RUNS_PER_DAY simulations into it, each with:
#   - a random hour within that day
#   - a random ocean start position drawn from that sector's box
#   - a random oil type and spill volume
#   - a random spill type (instantaneous with a random 50-100 m seed
#     radius, or continuous over a random 1-2 day release)
#
# Total runs = len(SECTORS_TO_RUN) x 365 (or 366) x RUNS_PER_DAY.
# NOTE: with the defaults above (2 sectors x 365 days x 10 runs) that is
# 7,300 simulations - expect this to take a long time to complete. Reduce
# SECTORS_TO_RUN or RUNS_PER_DAY first if you just want to test the flow.


def main():
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

    all_oil_type_names = all_oil_types(OIL_SCENARIOS)
    validate_oil_types(all_oil_type_names)
    oil_density_cache = build_oil_density_cache(all_oil_type_names)
    print("Oil density cache (kg/m3):", oil_density_cache)

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
        sector_csv_dir = os.path.join(OUTPUT_CSV_DIR, sector_tag)
        sector_png_dir = os.path.join(OUTPUT_IMAGE_DIR, sector_tag)

        for day in all_dates_in_year(YEAR):
            season_name = MONTH_TO_SEASON[day.month]
            day_folder_name = f"{season_name}_{day.strftime('%Y-%m-%d')}"

            day_csv_dir = os.path.join(sector_csv_dir, day_folder_name)
            day_png_dir = os.path.join(sector_png_dir, day_folder_name)
            os.makedirs(day_csv_dir, exist_ok=True)
            os.makedirs(day_png_dir, exist_ok=True)

            for run_number in range(1, RUNS_PER_DAY + 1):
                hour = random.randint(0, 23)
                start_time = datetime(day.year, day.month, day.day, hour)
                end_time = start_time + timedelta(days=SIM_DURATION_DAYS)

                oil_category = random.choice(list(OIL_SCENARIOS.keys()))
                scenario = OIL_SCENARIOS[oil_category]
                oil_type = random.choice(scenario["oil_types"])
                oil_mass_tonnes = random.uniform(scenario["tonnes_min"], scenario["tonnes_max"])
                oil_mass_kg = oil_mass_tonnes * 1000.0
                # approximate volume, from the real cached ADIOS density,
                # used only to give seed_elements() a realistic flow rate
                oil_volume_m3 = oil_mass_kg / oil_density_cache[oil_type]

                start_lon, start_lat = random_ocean_point_in_box(spill_box)

                spill_type = random.choice(SPILL_TYPES)
                if spill_type == "instantaneous":
                    spill_duration_hours = 0
                    seed_radius_meters = random.uniform(
                        INSTANTANEOUS_RADIUS_MIN_M, INSTANTANEOUS_RADIUS_MAX_M
                    )
                else:
                    spill_duration_hours = random.choice(CONTINUOUS_DURATION_CHOICES_HOURS)
                    seed_radius_meters = 0

                oil_tag = oil_type.replace(" ", "_")
                pos_tag = f"{start_lat:.3f}N_{start_lon:.3f}E"
                base_name = (
                    f"run{run_number}_{start_time.strftime('%H%M')}_{pos_tag}"
                    f"_{oil_category}_{oil_tag}_{oil_mass_tonnes:.0f}t_{spill_type}"
                )

                csv_path = os.path.join(day_csv_dir, f"{base_name}.csv")
                png_path = os.path.join(day_png_dir, f"{base_name}.png")

                print(
                    f"Running {sector_name}/{day_folder_name}/{base_name}: "
                    f"start={start_time}, position=({start_lon:.4f}, {start_lat:.4f}), "
                    f"category={oil_category}, oil={oil_type}, "
                    f"mass={oil_mass_tonnes:,.0f} tonnes, "
                    f"spill_type={spill_type}, "
                    f"spill_duration_hours={spill_duration_hours}, "
                    f"seed_radius_meters={seed_radius_meters:.1f}"
                )

                run_simulation_status = "ok"
                run_simulation_error = ""
                try:
                    run_simulation(
                        readers,
                        start_time,
                        end_time,
                        start_lon,
                        start_lat,
                        oil_type,
                        oil_mass_tonnes,
                        oil_volume_m3,
                        spill_duration_hours,
                        seed_radius_meters,
                        sector_name,
                        spill_type,
                        csv_path,
                        png_path,
                    )
                except Exception as exc:
                    # Log the failure and move on to the next run instead
                    # of aborting the whole batch. Also make sure any
                    # partially-created figures from this run are released
                    # before continuing.
                    run_simulation_status = "failed"
                    run_simulation_error = f"{type(exc).__name__}: {exc}"
                    print(
                        f"  !! Run {base_name} FAILED - continuing with next run.\n"
                        f"     {run_simulation_error}"
                    )
                    traceback.print_exc()
                    plt.close("all")

                log_rows.append(
                    {
                        "sector": sector_name,
                        "season": season_name,
                        "date": day.strftime("%Y-%m-%d"),
                        "run": run_number,
                        "start_time": start_time,
                        "end_time": end_time,
                        "start_lon": start_lon,
                        "start_lat": start_lat,
                        "oil_category": oil_category,
                        "oil_type": oil_type,
                        "oil_mass_tonnes": oil_mass_tonnes,
                        "spill_type": spill_type,
                        "spill_duration_hours": spill_duration_hours,
                        "seed_radius_meters": seed_radius_meters,
                        "status": run_simulation_status,
                        "error": run_simulation_error,
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

    failed_count = sum(1 for row in log_rows if row["status"] == "failed")
    ok_count = len(log_rows) - failed_count
    print(f"  {ok_count} succeeded, {failed_count} failed (see 'status'/'error' columns in the summary).")


if __name__ == "__main__":
    main()