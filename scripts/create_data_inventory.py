#!/usr/bin/env python3
"""Create a lightweight inventory of hyperspectral BIP scenes.

The script scans a root directory for ENVI BIP image files and writes a CSV
inventory without loading raster pixel data into memory.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


CSV_COLUMNS = [
    "flight_id",
    "date",
    "camera",
    "hdr_path",
    "image_path",
    "reflectance_status",
    "georeferenced",
    "crs",
    "notes",
]

DATE_PATTERNS = [
    re.compile(r"(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)"),
    re.compile(r"(\d{2})[-_]?([01]\d)[-_]?([0-3]\d)"),
]

CAMERA_PATTERNS = {
    "pika_ir_l": re.compile(r"pika[-_ ]?ir[-_ ]?l", re.IGNORECASE),
    "pika_l": re.compile(r"pika[-_ ]?l", re.IGNORECASE),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Root directory to scan.")
    parser.add_argument("--output", required=True, type=Path, help="Output inventory CSV path.")
    return parser.parse_args()


def normalize_date(text: str) -> str:
    """Extract normalized ISO date from text if present."""
    for pattern in DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            y, m, d = match.groups()
            if len(y) == 2:
                y = f"20{y}"
            return f"{y}-{m}-{d}"
    return ""


def infer_camera(text: str) -> str:
    """Infer camera family from path or file stem."""
    lowered = text.lower()
    for camera, pattern in CAMERA_PATTERNS.items():
        if pattern.search(lowered):
            return camera
    return "unknown"


def read_hdr_metadata(hdr_path: Path) -> dict[str, str]:
    """Parse minimal ENVI header metadata as key-value strings."""
    metadata: dict[str, str] = {}
    if not hdr_path.exists():
        return metadata

    for line in hdr_path.read_text(errors="ignore").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        metadata[key.strip().lower()] = value.strip()
    return metadata


def infer_flight_id(image_path: Path, root: Path) -> str:
    """Infer flight id from nearest parent directory under root."""
    try:
        relative = image_path.relative_to(root)
        if len(relative.parts) > 1:
            return relative.parts[0]
    except ValueError:
        pass
    return image_path.parent.name or image_path.stem


def inventory_row(image_path: Path, root: Path) -> dict[str, str]:
    """Build one inventory row from a BIP image path."""
    hdr_path = image_path.with_suffix(".hdr")
    metadata = read_hdr_metadata(hdr_path)

    path_text = str(image_path)
    reflectance_status = "reflectance" if re.search(r"refl|reflectance", path_text, re.IGNORECASE) else "unknown"

    crs = metadata.get("coordinate system string", "")
    map_info = metadata.get("map info", "")
    georeferenced = "yes" if crs or map_info else "no"

    date = normalize_date(path_text)
    camera = infer_camera(path_text)
    flight_id = infer_flight_id(image_path, root)

    notes: list[str] = []
    if not hdr_path.exists():
        notes.append("missing_hdr")
    if georeferenced == "no":
        notes.append("no_georef_metadata")

    return {
        "flight_id": flight_id,
        "date": date,
        "camera": camera,
        "hdr_path": str(hdr_path) if hdr_path.exists() else "",
        "image_path": str(image_path),
        "reflectance_status": reflectance_status,
        "georeferenced": georeferenced,
        "crs": crs,
        "notes": ";".join(notes),
    }


def find_bip_images(root: Path) -> list[Path]:
    """Find .bip images recursively under root."""
    return sorted(path for path in root.rglob("*.bip") if path.is_file())


def write_inventory(rows: list[dict[str, str]], output_path: Path) -> None:
    """Write inventory CSV with fixed schema."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()

    rows = [inventory_row(image_path, root) for image_path in find_bip_images(root)]
    write_inventory(rows, output)


if __name__ == "__main__":
    main()
