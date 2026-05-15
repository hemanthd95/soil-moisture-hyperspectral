"""Utilities for reading and aggregating flight-level ground-truth soil moisture CSV files."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

GROUNDTRUTH_COLUMNS = {
    "Logger Serial Number": "logger_serial_number",
    "Log ID": "log_id",
    "Address": "sensor_id",
    "Command": "command",
    "Soil Moisture": "soil_moisture",
    "Soil Temperature (°C)": "soil_temperature_c",
    "Permittivity": "permittivity",
    "Bulk Conductivity (µS/cm)": "bulk_conductivity_us_cm",
    "Pore Conductivity (µS/cm)": "pore_conductivity_us_cm",
    "Timestamp (EST)": "timestamp_est",
    "SDI_12_Address": "sensor_id",
    "Longitude": "longitude",
    "Latitude": "latitude",
    "Easting": "easting",
    "Northing": "northing",
}

FLIGHT_NAME_PATTERN = re.compile(
    r"SM_(?P<flight_id>F\d+?)_(?P<camera>[A-Za-z0-9]+?)_(?P<m>\d{1,2})_(?P<d>\d{1,2})_(?P<y>\d{2,4})",
    re.IGNORECASE,
)


def infer_flight_metadata(file_path: str | Path) -> dict[str, str]:
    """Infer flight id, camera, and date from a ground-truth filename."""
    stem = Path(file_path).stem
    match = FLIGHT_NAME_PATTERN.search(stem)
    if not match:
        return {"flight_id": "", "camera": "", "date": ""}

    year = match.group("y")
    if len(year) == 2:
        year = f"20{year}"

    date = f"{int(year):04d}-{int(match.group('m')):02d}-{int(match.group('d')):02d}"
    return {
        "flight_id": match.group("flight_id").upper(),
        "camera": match.group("camera"),
        "date": date,
    }


def read_groundtruth_csv(path: str | Path, timezone: str = "America/New_York") -> pd.DataFrame:
    """Read one raw ground-truth CSV and normalize columns and timestamp."""
    src = Path(path)
    df = pd.read_csv(src)
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    # Prefer SDI_12_Address as the sensor ID if available.
    # # Keep Address separately to avoid duplicate sensor_id columns.
    if "SDI_12_Address" in df.columns:
        df = df.rename(columns={"SDI_12_Address": "sensor_id"})
    elif "Address" in df.columns:
        df = df.rename(columns={"Address": "sensor_id"})
    COLUMN_ALIASES = {
        "Logger Serial Number": "logger_serial_number",
        "Log ID": "log_id",
        "Command": "command",
        "Soil Moisture": "soil_moisture",
        "Soil Temperature (°C)": "soil_temperature_c",
        "Permittivity": "permittivity",
        "Bulk Conductivity (µS/cm)": "bulk_conductivity_us_cm",
        "Pore Conductivity (µS/cm)": "pore_conductivity_us_cm",
        "Timestamp (EST)": "timestamp_est",
        "Longitude": "longitude",
        "Latitude": "latitude",
        "Easting": "easting",
        "Northing": "northing",
        }
    df = df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns})
    # Remove any accidental duplicate columns, keeping the first occurrence
    df = df.loc[:, ~df.columns.duplicated()].copy()
    #df = df.rename(columns=GROUNDTRUTH_COLUMNS)

    required = [
        "sensor_id",
        "timestamp_est",
        "soil_moisture",
        "soil_temperature_c",
        "permittivity",
        "bulk_conductivity_us_cm",
        "pore_conductivity_us_cm",
        "longitude",
        "latitude",
        "easting",
        "northing",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {src.name}: {missing}")

    for c in [
        "soil_moisture",
        "soil_temperature_c",
        "permittivity",
        "bulk_conductivity_us_cm",
        "pore_conductivity_us_cm",
        "longitude",
        "latitude",
        "easting",
        "northing",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    ts = pd.to_datetime(df["timestamp_est"], errors="coerce")
    #df["timestamp_est"] = ts.dt.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT")
    ts = pd.to_datetime(df["timestamp_est"], errors="coerce")
    if getattr(ts.dt, "tz", None) is None:
        df["timestamp_est"] = ts.dt.tz_localize(
        timezone,
        ambiguous="NaT",
        nonexistent="NaT"
    )
    else:
        df["timestamp_est"] = ts.dt.tz_convert(timezone)

    meta = infer_flight_metadata(src)
    df["flight_id"] = meta["flight_id"]
    df["camera"] = meta["camera"]
    df["date"] = meta["date"]
    df["source_file"] = str(src)
    return df


def read_groundtruth_folder(folder: str | Path) -> pd.DataFrame:
    """Read all CSV files from folder and concatenate them."""
    root = Path(folder)
    files = sorted(root.glob("*.csv"))
    if not files:
        return pd.DataFrame()
    frames = [read_groundtruth_csv(path) for path in files]
    return pd.concat(frames, ignore_index=True)


def build_groundtruth_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw measurements to one row per sensor per flight."""
    if raw_df.empty:
        return pd.DataFrame()

    group_cols = ["flight_id", "camera", "date", "sensor_id"]
    value_cols = [
        "soil_moisture",
        "soil_temperature_c",
        "permittivity",
        "bulk_conductivity_us_cm",
        "pore_conductivity_us_cm",
    ]

    agg_map: dict[str, list[str]] = {c: ["mean", "std"] for c in value_cols}
    agg_map.update(
        {
            "longitude": "first",
            "latitude": "first",
            "easting": "first",
            "northing": "first",
            "timestamp_est": ["min", "max", "count"],
        }
    )

    out = raw_df.groupby(group_cols, dropna=False).agg(agg_map)
    out.columns = ["_".join(col) if isinstance(col, tuple) else col for col in out.columns.to_flat_index()]
    out = out.reset_index()

    out = out.rename(
        columns={
            "longitude_first": "longitude",
            "latitude_first": "latitude",
            "easting_first": "easting",
            "northing_first": "northing",
            "timestamp_est_min": "flight_start_time_est",
            "timestamp_est_max": "flight_end_time_est",
            "timestamp_est_count": "n_observations",
        }
    )
    return out
