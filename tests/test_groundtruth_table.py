from pathlib import Path

import pandas as pd

from src.io.read_groundtruth import build_groundtruth_table, infer_flight_metadata, read_groundtruth_csv, read_groundtruth_folder


def test_infer_flight_metadata_from_filename():
    meta = infer_flight_metadata("SM_F1_PikaL_11_7_25_complete.csv")
    assert meta["flight_id"] == "F1"
    assert meta["camera"] == "PikaL"
    assert meta["date"] == "2025-11-07"


def test_read_groundtruth_csv_and_aggregate(tmp_path: Path):
    csv_path = tmp_path / "SM_F1_PikaL_11_7_25_complete.csv"
    df = pd.DataFrame(
        {
            "Logger Serial Number": ["A", "A", "B"],
            "Log ID": [1, 2, 3],
            "Address": ["x", "x", "y"],
            "Command": ["R", "R", "R"],
            "Soil Moisture": [0.10, 0.14, 0.20],
            "Soil Temperature (°C)": [20.0, 22.0, 19.0],
            "Permittivity": [5.0, 6.0, 7.0],
            "Bulk Conductivity (µS/cm)": [100.0, 120.0, 130.0],
            "Pore Conductivity (µS/cm)": [80.0, 90.0, 110.0],
            "Timestamp (EST)": ["2025-11-07 10:00:00", "2025-11-07 10:05:00", "2025-11-07 10:10:00"],
            "SDI_12_Address": ["S1", "S1", "S2"],
            "Longitude": [-81.0, -81.0, -81.1],
            "Latitude": [33.0, 33.0, 33.1],
            "Easting": [500000, 500000, 500010],
            "Northing": [3600000, 3600000, 3600010],
        }
    )
    df.to_csv(csv_path, index=False)

    raw = read_groundtruth_csv(csv_path)
    assert str(raw["timestamp_est"].dt.tz) == "America/New_York"

    agg = build_groundtruth_table(raw)
    assert len(agg) == 2

    s1 = agg.loc[agg["sensor_id"] == "S1"].iloc[0]
    assert s1["soil_moisture_mean"] == 0.12
    assert int(s1["n_observations"]) == 2
    assert pd.notna(s1["flight_start_time_est"])
    assert pd.notna(s1["flight_end_time_est"])


def test_read_groundtruth_folder_many_files(tmp_path: Path):
    a = tmp_path / "SM_F2_PikaIRL_11_8_2025_complete.csv"
    b = tmp_path / "SM_F3_PikaL_11_9_2025_complete.csv"

    for path in [a, b]:
        pd.DataFrame(
            {
                "Logger Serial Number": ["A"],
                "Log ID": [1],
                "Address": ["x"],
                "Command": ["R"],
                "Soil Moisture": [0.10],
                "Soil Temperature (°C)": [20.0],
                "Permittivity": [5.0],
                "Bulk Conductivity (µS/cm)": [100.0],
                "Pore Conductivity (µS/cm)": [80.0],
                "Timestamp (EST)": ["2025-11-07 10:00:00"],
                "SDI_12_Address": ["S1"],
                "Longitude": [-81.0],
                "Latitude": [33.0],
                "Easting": [500000],
                "Northing": [3600000],
            }
        ).to_csv(path, index=False)

    raw = read_groundtruth_folder(tmp_path)
    assert len(raw) == 2
