import csv
import subprocess
import sys
from pathlib import Path


def test_create_data_inventory_script(tmp_path: Path):
    root = tmp_path / "data"
    flight = root / "flight_20260401"
    flight.mkdir(parents=True)

    image = flight / "Pika_L_reflectance_scene.bip"
    image.write_bytes(b"")

    hdr = flight / "Pika_L_reflectance_scene.hdr"
    hdr.write_text(
        "ENVI\n"
        "map info = {UTM, 1.0, 1.0, 500000.0, 3600000.0, 1.0, 1.0, 17, North, WGS-84}\n"
        "coordinate system string = PROJCS[\"WGS 84 / UTM zone 17N\"]\n",
        encoding="utf-8",
    )

    output = tmp_path / "inventory.csv"

    subprocess.run(
        [
            sys.executable,
            "scripts/create_data_inventory.py",
            "--root",
            str(root),
            "--output",
            str(output),
        ],
        check=True,
    )

    with output.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    row = rows[0]
    assert row["camera"] == "pika_l"
    assert row["date"] == "2026-04-01"
    assert row["reflectance_status"] == "reflectance"
    assert row["georeferenced"] == "yes"
    assert row["image_path"].endswith(".bip")
