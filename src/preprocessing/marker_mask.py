"""Marker/white paint mask utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def white_marker_mask(cube: np.ndarray, wavelengths: list[float], config: dict[str, Any]) -> np.ndarray:
    """Detect white-paint/marker pixels.

    Rule: high visible reflectance AND low spectral smoothness (high roughness)
    in visible range, with thresholds loaded from YAML config.
    """
    if cube.ndim != 3:
        raise ValueError("cube must be 3D: (bands, rows, cols)")

    w = np.asarray(wavelengths)
    vis_idx = np.where((w >= 450.0) & (w <= 700.0))[0]
    if vis_idx.size < 3:
        vis_idx = np.arange(min(cube.shape[0], 5))

    vis_cube = cube[vis_idx]
    vis_mean = np.nanmean(vis_cube, axis=0)

    # Smoothness proxy: mean absolute second spectral difference.
    d2 = np.diff(vis_cube, n=2, axis=0)
    roughness = np.nanmean(np.abs(d2), axis=0) if d2.size else np.zeros_like(vis_mean)

    vis_thr = float(config["thresholds"]["white_visible_min"])
    rough_thr = float(config["thresholds"]["white_roughness_min"])
    return (vis_mean >= vis_thr) & (roughness >= rough_thr)


def plot_marker_diagnostics(
    vis_mean: np.ndarray,
    roughness: np.ndarray,
    marker: np.ndarray,
    output_path: str | Path,
) -> None:
    """Save diagnostic plot for white marker detection."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(vis_mean, cmap="viridis")
    axes[0].set_title("Visible mean")
    axes[1].imshow(roughness, cmap="magma")
    axes[1].set_title("Visible roughness")
    axes[2].imshow(marker, cmap="binary")
    axes[2].set_title("White marker mask")
    for ax in axes:
        ax.set_axis_off()
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
