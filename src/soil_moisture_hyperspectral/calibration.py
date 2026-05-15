"""Calibration utilities for sensor-linked modeling."""

import numpy as np


def split_train_test_by_index(n_samples: int, test_fraction: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic index-based train/test split arrays."""
    n_test = max(1, int(round(n_samples * test_fraction)))
    idx = np.arange(n_samples)
    return idx[:-n_test], idx[-n_test:]


def fit_linear_calibration(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit a simple linear relation y = a*x + b as placeholder calibration."""
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)
