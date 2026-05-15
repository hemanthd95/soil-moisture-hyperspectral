"""Mask utilities for vegetation and shadow/dark pixel screening."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_mask_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML config containing mask thresholds and options."""
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def nearest_band_index(wavelengths: list[float], target_nm: float) -> int:
    """Return zero-based band index nearest to target wavelength."""
    arr = np.asarray(wavelengths, dtype=float)
    if arr.size == 0:
        raise ValueError("wavelengths must not be empty")
    return int(np.argmin(np.abs(arr - float(target_nm))))


def compute_ndvi_from_cube(cube: np.ndarray, wavelengths: list[float]) -> np.ndarray:
    """Compute NDVI using nearest red (~670nm) and NIR (~800nm) bands.

    Parameters
    ----------
    cube
        Array of shape (bands, rows, cols).
    wavelengths
        Band-center wavelengths (nm), length must equal band count.
    """
    if cube.ndim != 3:
        raise ValueError("cube must be 3D: (bands, rows, cols)")
    if cube.shape[0] != len(wavelengths):
        raise ValueError("cube band count must match wavelengths length")

    red_idx = nearest_band_index(wavelengths, 670.0)
    nir_idx = nearest_band_index(wavelengths, 800.0)
    red = cube[red_idx]
    nir = cube[nir_idx]
    eps = 1e-6
    return (nir - red) / (nir + red + eps)


def vegetation_mask(cube: np.ndarray, wavelengths: list[float], config: dict[str, Any]) -> np.ndarray:
    """Create vegetation mask using NDVI threshold from config."""
    ndvi_thr = float(config["thresholds"]["ndvi_veg_min"])
    ndvi = compute_ndvi_from_cube(cube, wavelengths)
    return ndvi >= ndvi_thr


def shadow_mask(cube: np.ndarray, wavelengths: list[float], config: dict[str, Any]) -> np.ndarray:
    """Create dark/shadow mask from configurable reflectance threshold.

    Uses mean visible reflectance over 450-700 nm.
    """
    thr = float(config["thresholds"]["dark_reflectance_max"])
    w = np.asarray(wavelengths)
    vis_idx = np.where((w >= 450.0) & (w <= 700.0))[0]
    if vis_idx.size == 0:
        vis_idx = np.arange(min(3, cube.shape[0]))
    vis_mean = np.nanmean(cube[vis_idx], axis=0)
    return vis_mean <= thr


def plot_mask_diagnostics(ndvi: np.ndarray, veg: np.ndarray, dark: np.ndarray, output_path: str | Path) -> None:
    """Save simple diagnostic plot showing NDVI and generated masks."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(ndvi, cmap="RdYlGn")
    axes[0].set_title("NDVI")
    axes[1].imshow(veg, cmap="Greens")
    axes[1].set_title("Vegetation mask")
    axes[2].imshow(dark, cmap="gray")
    axes[2].set_title("Shadow mask")
    for ax in axes:
        ax.set_axis_off()
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
