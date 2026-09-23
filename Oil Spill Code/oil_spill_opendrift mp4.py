# LOADING LIBRARIES

import csv
import math
import os
import random
import traceback
from datetime import datetime, timedelta

# Force non-interactive Agg backend before matplotlib/opendrift imports
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Adjust figure size, resolution, and top margin to prevent title cutoff
plt.rcParams["figure.figsize"] = (11, 7.64)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["figure.subplot.top"] = 0.78     # Reserves top space for multi-line titles
plt.rcParams["figure.subplot.bottom"] = 0.08
plt.rcParams["figure.subplot.left"] = 0.08
plt.rcParams["figure.subplot.right"] = 0.95
plt.rcParams["axes.titlesize"] = 10           # Clean title font size

from global_land_mask import globe
import numpy as np
from opendrift.models.openoil import OpenOil
from opendrift.readers import reader_netCDF_CF_generic
import xarray as xr

# ---------------------------------------------------------
# INPUT FILES

WIND_FILE = r"E:\University\Applied Oceanography\Dissertation\Data\Oil Spill Code Data\Wind2.grib"
WAVE_FILE = r"E:\University\Applied Oceanography\Dissertation\Data\Oil Spill Code Data\Waves2.nc"
CURRENTS_FILE = r"E:\University\Applied Oceanography\Dissertation\Data\Oil Spill Code Data\Currents2.nc"
COORDINATES_FILE = r"E:\University\Applied Oceanography\Dissertation\Data\Oil Spill Code Data\coordinates.csv"

SECTORS_TO_RUN = ["Malta North"]

# ---------------------------------------------------------
# OIL SPILL SCENARIOS

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

OUTPUT_GIF_DIR = r"E:\University\Applied Oceanography\Dissertation\Results\Oilspills Mp4"

# ---------------------------------------------------------
# SIMULATION PARAMETERS

YEAR = 2025
SIM_DURATION_DAYS = 4

# Number of random days to sample from the year
NUM_RANDOM_DAYS = 15

# How many random runs to execute per selected day
RUNS_PER_DAY = 1

SEASONS = {
    "winter": [12, 1, 2],
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "autumn": [9, 10, 11],
}

MONTH_TO_SEASON = {
    month: season_name for season_name, months in SEASONS.items() for month in months
}

RANDOM_SEED = None

SPILL_TYPES = ["instantaneous", "continuous"]
INSTANTANEOUS_RADIUS_MIN_M = 50
INSTANTANEOUS_RADIUS_MAX_M = 100
CONTINUOUS_DURATION_CHOICES_HOURS = [24, 48]

NUMBER_OF_PARTICLES = 500
MODEL_TIME_STEP_SECONDS = 900
OUTPUT_EVERY_SECONDS = 3600  # 1 hour steps

USE_WAVE_STOKES_DRIFT = True
USE_WAVE_MIXING = True
USE_OIL_WEATHERING = True

# Plot extent
PLOT_LON_MIN = 12.96
PLOT_LON_MAX = 15.84
PLOT_LAT_MIN = 35.5
PLOT_LAT_MAX = 37.110

# Animation speed
ANIMATION_FPS = 5

# ---------------------------------------------------------
# SPILL-BOX LOADING


def load_spill_boxes(path):
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
    lon = random.uniform(box["lon_min"], box["lon_max"])
    lat = random.uniform(box["lat_min"], box["lat_max"])
    return lon, lat


def random_ocean_point_in_box(box, max_attempts=500):
    for _ in range(max_attempts):
        lon, lat = random_point_in_box(box)
        if not globe.is_land(lat, lon):
            return lon, lat
    raise RuntimeError(
        f"Could not find an ocean point in box {box} after {max_attempts} attempts."
    )


# ---------------------------------------------------------
# OIL TYPE VALIDATION / DENSITY LOOKUP


def all_oil_types(oil_scenarios):
    types = []
    for scenario in oil_scenarios.values():
        types.extend(scenario["oil_types"])
    return types


def validate_oil_types(oil_types):
    checker = OpenOil(loglevel=50)
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
        )


def build_oil_density_cache(oil_types):
    cache = {}
    for oil_type in oil_types:
        probe = OpenOil(loglevel=50)
        probe.set_config("seed:oil_type", oil_type)
        for var in (
            "x_sea_water_velocity",
            "y_sea_water_velocity",
            "x_wind",
            "y_wind",
        ):
            probe.set_config(f"environment:fallback:{var}", 0)
        probe.seed_elements(
            lon=0, lat=0, radius=0, number=1, time=datetime(2000, 1, 1), z=0
        )
        cache[oil_type] = float(probe.elements_scheduled.density)
    return cache


# ---------------------------------------------------------
# DATE HELPER


def all_dates_in_year(year):
    current = datetime(year, 1, 1)
    end = datetime(year + 1, 1, 1)
    while current < end:
        yield current
        current += timedelta(days=1)


# ---------------------------------------------------------
# READERS


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
# SINGLE SIMULATION RUN WITH GIF ANIMATION


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
    gif_path,
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

    oil_mass_kg = oil_mass_tonnes * 1000.0
    model.elements_scheduled.mass_oil = oil_mass_kg / NUMBER_OF_PARTICLES

    model.run(
        duration=end_time - start_time,
        time_step=MODEL_TIME_STEP_SECONDS,
        time_step_output=OUTPUT_EVERY_SECONDS,
    )

    if spill_type == "instantaneous":
        spill_desc = f"Instantaneous release, seed radius {seed_radius_meters:.0f} m"
    else:
        spill_desc = f"Continuous release, duration {spill_duration_hours:.0f} h"

    # Header title string formatted across lines
    title = (
        f"{sector_name} | {start_time.strftime('%Y-%m-%d %H:%M')} UTC | {oil_type} ({oil_mass_tonnes:,.0f} tonnes)\n"
        f"{spill_desc} | Start: {start_lat:.3f}N, {start_lon:.3f}E | Simulated {SIM_DURATION_DAYS} days"
    )

    # Render animation: legend=None auto-generates particle status legend box
    model.animation(
        filename=gif_path,
        fast=True,
        drift_tracks=True,     # Draws grey trajectory lines trailing behind particles
        corners=[PLOT_LON_MIN, PLOT_LON_MAX, PLOT_LAT_MIN, PLOT_LAT_MAX],
        title=title,
        legend=None,           # OpenDrift auto-builds standard status legend box
        fps=ANIMATION_FPS,
    )

    plt.close("all")


# ---------------------------------------------------------
# MAIN PROGRAM


def main():
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    os.makedirs(OUTPUT_GIF_DIR, exist_ok=True)

    all_oil_type_names = all_oil_types(OIL_SCENARIOS)
    validate_oil_types(all_oil_type_names)
    oil_density_cache = build_oil_density_cache(all_oil_type_names)
    print("Oil density cache (kg/m3):", oil_density_cache)

    all_boxes = load_spill_boxes(COORDINATES_FILE)

    # Sample random calendar days
    all_year_days = list(all_dates_in_year(YEAR))
    sampled_days = sorted(random.sample(all_year_days, NUM_RANDOM_DAYS))
    print(f"\nSampled {len(sampled_days)} random days for simulation GIF export.\n")

    readers = build_readers()
    log_rows = []

    for sector_name in SECTORS_TO_RUN:
        spill_box = get_box_by_name(all_boxes, sector_name)
        sector_tag = sector_name.strip().replace(" ", "_")
        sector_gif_dir = os.path.join(OUTPUT_GIF_DIR, sector_tag)

        for day in sampled_days:
            season_name = MONTH_TO_SEASON[day.month]
            day_folder_name = f"{season_name}_{day.strftime('%Y-%m-%d')}"
            day_gif_dir = os.path.join(sector_gif_dir, day_folder_name)
            os.makedirs(day_gif_dir, exist_ok=True)

            for run_number in range(1, RUNS_PER_DAY + 1):
                hour = random.randint(0, 23)
                start_time = datetime(day.year, day.month, day.day, hour)
                end_time = start_time + timedelta(days=SIM_DURATION_DAYS)

                oil_category = random.choice(list(OIL_SCENARIOS.keys()))
                scenario = OIL_SCENARIOS[oil_category]
                oil_type = random.choice(scenario["oil_types"])
                oil_mass_tonnes = random.uniform(
                    scenario["tonnes_min"], scenario["tonnes_max"]
                )
                oil_mass_kg = oil_mass_tonnes * 1000.0
                oil_volume_m3 = oil_mass_kg / oil_density_cache[oil_type]

                start_lon, start_lat = random_ocean_point_in_box(spill_box)

                spill_type = random.choice(SPILL_TYPES)
                if spill_type == "instantaneous":
                    spill_duration_hours = 0
                    seed_radius_meters = random.uniform(
                        INSTANTANEOUS_RADIUS_MIN_M, INSTANTANEOUS_RADIUS_MAX_M
                    )
                else:
                    spill_duration_hours = random.choice(
                        CONTINUOUS_DURATION_CHOICES_HOURS
                    )
                    seed_radius_meters = 0

                oil_tag = oil_type.replace(" ", "_")
                pos_tag = f"{start_lat:.3f}N_{start_lon:.3f}E"
                base_name = (
                    f"run{run_number}_{start_time.strftime('%H%M')}_{pos_tag}"
                    f"_{oil_category}_{oil_tag}_{oil_mass_tonnes:.0f}t_{spill_type}"
                )

                gif_path = os.path.join(day_gif_dir, f"{base_name}.gif")

                print(
                    f"Simulating & rendering GIF for {sector_name}/{day_folder_name}/{base_name}: "
                    f"start={start_time}, position=({start_lon:.4f}, {start_lat:.4f})"
                )

                run_status = "ok"
                run_error = ""
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
                        gif_path,
                    )
                except Exception as exc:
                    run_status = "failed"
                    run_error = f"{type(exc).__name__}: {exc}"
                    print(
                        f"  !! Run {base_name} FAILED - skipping.\n     {run_error}"
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
                        "status": run_status,
                        "error": run_error,
                        "gif_file": gif_path,
                    }
                )

    log_path = os.path.join(OUTPUT_GIF_DIR, "run_summary.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"\nAll runs complete. GIF log summary written to {log_path}")


if __name__ == "__main__":
    main()