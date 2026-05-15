"""Window-based sensor spectra extraction for georeferenced hyperspectral rasters."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _normalize_radii(buffer_radii_m: float | Iterable[float]) -> list[float]:
    if isinstance(buffer_radii_m, (int, float)):
        return [float(buffer_radii_m)]
    radii = [float(v) for v in buffer_radii_m]
    if not radii:
        raise ValueError("At least one buffer radius is required.")
    return radii


def _build_valid_mask(
    cube: np.ndarray,
    nodata: float | None,
    vegetation_mask: np.ndarray | None,
    shadow_mask: np.ndarray | None,
    white_paint_mask: np.ndarray | None,
    nodata_mask: np.ndarray | None,
) -> np.ndarray:
    rows, cols = cube.shape[1], cube.shape[2]
    valid = np.ones((rows, cols), dtype=bool)

    if nodata is not None:
        valid &= ~np.any(np.isclose(cube, nodata), axis=0)
    valid &= np.all(np.isfinite(cube), axis=0)

    for mask in [vegetation_mask, shadow_mask, white_paint_mask, nodata_mask]:
        if mask is not None:
            valid &= ~mask.astype(bool)
    return valid


def extract_sensor_spectra(
    raster_path: str | Path,
    wavelengths: list[float],
    sensors: pd.DataFrame,
    buffer_radii_m: float | Iterable[float],
    *,
    sensor_id_col: str = "sensor_id",
    x_col: str = "easting",
    y_col: str = "northing",
    flight_id: str = "",
    poor_valid_threshold: int = 5,
    vegetation_mask: np.ndarray | None = None,
    shadow_mask: np.ndarray | None = None,
    white_paint_mask: np.ndarray | None = None,
    nodata_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """Extract per-wavelength stats around each sensor using raster windows.

    This function never reads the full raster cube and instead uses small windows
    per sensor and per radius.
    """
    import rasterio
    from rasterio.windows import Window

    radii = _normalize_radii(buffer_radii_m)
    if len(wavelengths) == 0:
        raise ValueError("wavelengths must be non-empty.")

    rows_out: list[dict[str, object]] = []

    with rasterio.open(raster_path) as src:
        if src.count != len(wavelengths):
            raise ValueError("Band count and wavelength list length do not match.")

        xres = abs(src.transform.a)
        yres = abs(src.transform.e)
        nodata = src.nodata

        for _, sensor in sensors.iterrows():
            sensor_id = sensor[sensor_id_col]
            x = float(sensor[x_col])
            y = float(sensor[y_col])
            pixel_row, pixel_col = src.index(x, y)

            for radius_m in radii:
                r_pix_x = max(1, int(np.ceil(radius_m / xres)))
                r_pix_y = max(1, int(np.ceil(radius_m / yres)))

                row_off = max(0, pixel_row - r_pix_y)
                col_off = max(0, pixel_col - r_pix_x)
                row_end = min(src.height, pixel_row + r_pix_y + 1)
                col_end = min(src.width, pixel_col + r_pix_x + 1)

                win = Window(
                    col_off=col_off,
                    row_off=row_off,
                    width=col_end - col_off,
                    height=row_end - row_off,
                )
                cube = src.read(window=win)

                valid_mask = _build_valid_mask(
                    cube,
                    nodata=nodata,
                    vegetation_mask=vegetation_mask[row_off:row_end, col_off:col_end] if vegetation_mask is not None else None,
                    shadow_mask=shadow_mask[row_off:row_end, col_off:col_end] if shadow_mask is not None else None,
                    white_paint_mask=white_paint_mask[row_off:row_end, col_off:col_end] if white_paint_mask is not None else None,
                    nodata_mask=nodata_mask[row_off:row_end, col_off:col_end] if nodata_mask is not None else None,
                )

                record: dict[str, object] = {
                    "flight_id": flight_id,
                    "sensor_id": sensor_id,
                    "buffer_radius_m": radius_m,
                    "pixel_row": int(pixel_row),
                    "pixel_col": int(pixel_col),
                    "valid_pixel_count": int(valid_mask.sum()),
                    "poor_valid": bool(valid_mask.sum() < poor_valid_threshold),
                }

                for band_idx, wl in enumerate(wavelengths):
                    vals = cube[band_idx][valid_mask]
                    base = f"wl_{int(round(wl))}"
                    if vals.size == 0:
                        record[f"{base}_mean"] = np.nan
                        record[f"{base}_median"] = np.nan
                        record[f"{base}_std"] = np.nan
                    else:
                        record[f"{base}_mean"] = float(np.nanmean(vals))
                        record[f"{base}_median"] = float(np.nanmedian(vals))
                        record[f"{base}_std"] = float(np.nanstd(vals, ddof=0))

                rows_out.append(record)

    return pd.DataFrame(rows_out)


def save_sensor_spectra_parquet(table: pd.DataFrame, output_path: str | Path) -> None:
    """Save extracted sensor spectra table to parquet."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
