"""Mapping/export scaffolding for soil moisture rasters."""

from pathlib import Path
import numpy as np


def create_moisture_map(prediction: np.ndarray) -> np.ndarray:
    """Return a clipped moisture map in [0, 1] as a placeholder output."""
    return np.clip(prediction, 0.0, 1.0)


def export_geotiff_placeholder(array: np.ndarray, output_path: str | Path) -> dict[str, object]:
    """Return placeholder metadata describing a planned GeoTIFF export."""
    return {
        "output_path": str(Path(output_path)),
        "shape": tuple(array.shape),
        "dtype": str(array.dtype),
        "written": False,
    }
