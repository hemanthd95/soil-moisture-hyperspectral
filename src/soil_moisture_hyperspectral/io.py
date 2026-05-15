"""Input/output helpers for hyperspectral soil moisture workflows."""

from pathlib import Path
from typing import Iterable


def validate_path(path: str | Path) -> Path:
    """Validate and return a normalized path.

    Parameters
    ----------
    path
        Input filesystem path.

    Returns
    -------
    Path
        Normalized ``Path`` object.
    """
    return Path(path).expanduser().resolve()


def build_sensor_table(sensor_ids: Iterable[str]) -> list[dict[str, str]]:
    """Create a lightweight sensor lookup table.

    Parameters
    ----------
    sensor_ids
        Iterable of sensor identifiers.

    Returns
    -------
    list[dict[str, str]]
        Placeholder sensor rows for downstream joins.
    """
    return [{"sensor_id": sid} for sid in sensor_ids]


def open_hyperspectral_cube(path: str | Path) -> dict[str, object]:
    """Return placeholder metadata for a hyperspectral cube.

    This function intentionally avoids loading full raster data and acts as a
    scaffold for future rasterio/xarray-backed IO.
    """
    resolved = validate_path(path)
    return {"path": str(resolved), "backend": "placeholder", "loaded": False}
