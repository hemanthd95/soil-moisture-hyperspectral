"""Preprocessing functions for spectral cleanup and masking."""

import numpy as np


def mask_invalid_pixels(cube: np.ndarray, invalid_value: float = np.nan) -> np.ndarray:
    """Mask invalid or non-finite pixels in a spectral cube.

    Parameters
    ----------
    cube
        Spectral cube shaped ``(bands, rows, cols)`` or ``(rows, cols, bands)``.
    invalid_value
        Value to assign where pixels are non-finite.

    Returns
    -------
    np.ndarray
        Cube with non-finite values replaced.
    """
    output = np.array(cube, copy=True)
    output[~np.isfinite(output)] = invalid_value
    return output


def compute_ndvi(nir: np.ndarray, red: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute NDVI from NIR and red bands."""
    return (nir - red) / (nir + red + eps)


def compute_str(swir: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute Shortwave Transform Ratio (STR) style index placeholder."""
    return ((1.0 - swir) ** 2) / (2.0 * swir + eps)
