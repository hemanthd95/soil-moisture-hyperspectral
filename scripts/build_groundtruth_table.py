#!/usr/bin/env python3
"""Build cleaned ground-truth table from per-flight CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.io.read_groundtruth import build_groundtruth_table, read_groundtruth_folder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Folder containing raw flight CSV files.")
    parser.add_argument("--output-prefix", required=True, type=Path, help="Output path prefix for .csv and .parquet.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = read_groundtruth_folder(args.input)
    table = build_groundtruth_table(raw)

    output_prefix = args.output_prefix.expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    csv_path = output_prefix.with_suffix(".csv")
    parquet_path = output_prefix.with_suffix(".parquet")

    table.to_csv(csv_path, index=False)
    table.to_parquet(parquet_path, index=False)


if __name__ == "__main__":
    main()
