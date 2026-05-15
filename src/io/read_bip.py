"""Memory-safe readers for large ENVI BIP hyperspectral datasets.

This module focuses on metadata inspection and selective reads only, so callers do
not need to load full cubes into memory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _parse_envi_header(hdr_path: Path) -> dict[str, str]:
    """Parse simple ENVI header key-value pairs.

    Notes
    -----
    This intentionally supports lightweight parsing for metadata extraction.
    """
    metadata: dict[str, str] = {}
    if not hdr_path.exists():
        return metadata

    for line in hdr_path.read_text(errors="ignore").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        metadata[key.strip().lower()] = value.strip()
    return metadata


def _parse_wavelengths(raw: str | None) -> list[float]:
    """Parse ENVI wavelength list from header text."""
    if not raw:
        return []
    cleaned = raw.replace("{", "").replace("}", "")
    wavelengths: list[float] = []
    for token in cleaned.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            wavelengths.append(float(token))
        except ValueError:
            continue
    return wavelengths


def read_bip_metadata(image_path: str | Path) -> dict[str, Any]:
    """Read BIP metadata using rasterio where possible, otherwise ENVI header.

    Returns keys: wavelengths, band_count, height, width, interleave, data_type,
    crs, geotransform.
    """
    image_path = Path(image_path)
    hdr_path = image_path.with_suffix(".hdr")
    hdr_meta = _parse_envi_header(hdr_path)

    result: dict[str, Any] = {
        "wavelengths": _parse_wavelengths(hdr_meta.get("wavelength")),
        "band_count": int(hdr_meta.get("bands", 0) or 0),
        "height": int(hdr_meta.get("lines", 0) or 0),
        "width": int(hdr_meta.get("samples", 0) or 0),
        "interleave": hdr_meta.get("interleave", ""),
        "data_type": hdr_meta.get("data type", ""),
        "crs": hdr_meta.get("coordinate system string", ""),
        "geotransform": hdr_meta.get("map info", ""),
        "backend": "header",
    }

    try:
        import rasterio

        with rasterio.open(image_path) as src:
            result.update(
                {
                    "band_count": src.count,
                    "height": src.height,
                    "width": src.width,
                    "data_type": src.dtypes[0] if src.dtypes else result["data_type"],
                    "crs": src.crs.to_string() if src.crs else result["crs"],
                    "geotransform": tuple(src.transform) if src.transform else result["geotransform"],
                    "backend": "rasterio",
                }
            )
    except Exception:
        pass

    return result


def read_selected_bands(image_path: str | Path, band_indices: list[int]) -> np.ndarray:
    """Read only selected 1-based band indices as a memory-safe subset array."""
    try:
        import rasterio

        with rasterio.open(image_path) as src:
            return src.read(indexes=band_indices)
    except Exception:
        try:
            from spectral import io as spectral_io

            img = spectral_io.envi.open(str(Path(image_path).with_suffix(".hdr")), str(image_path))
            arr = img.read_bands([idx - 1 for idx in band_indices])
            # spectral returns rows, cols, bands -> convert to bands, rows, cols
            return np.moveaxis(np.asarray(arr), -1, 0)
        except Exception as exc:
            raise RuntimeError(f"Unable to read selected bands from {image_path}: {exc}") from exc


def read_window(
    image_path: str | Path,
    *,
    row: int | None = None,
    col: int | None = None,
    x: float | None = None,
    y: float | None = None,
    half_size: int = 1,
    band_indices: list[int] | None = None,
) -> np.ndarray:
    """Read a small raster window around a pixel or map coordinate.

    Parameters
    ----------
    row, col
        Pixel coordinates (0-based).
    x, y
        Map coordinates; used when row/col are not provided.
    half_size
        Number of pixels around center in each direction. Window size equals
        ``(2*half_size + 1)``.
    band_indices
        Optional 1-based bands to read; defaults to all bands.
    """
    try:
        import rasterio
        from rasterio.windows import Window

        with rasterio.open(image_path) as src:
            if row is None or col is None:
                if x is None or y is None:
                    raise ValueError("Provide either (row, col) or (x, y).")
                row, col = src.index(x, y)

            win = Window(
                col_off=max(0, col - half_size),
                row_off=max(0, row - half_size),
                width=min(src.width - max(0, col - half_size), 2 * half_size + 1),
                height=min(src.height - max(0, row - half_size), 2 * half_size + 1),
            )
            return src.read(indexes=band_indices, window=win)
    except Exception as exc:
        raise RuntimeError(f"Windowed reads require rasterio support: {exc}") from exc
