"""MARMIT/MARMIT-2 calibration scaffolding."""

import numpy as np


def marmit_forward_model(reflectance: np.ndarray, theta: float = 0.5) -> np.ndarray:
    """Apply a placeholder MARMIT-style forward transform to reflectance."""
    return np.clip(reflectance * theta, 0.0, 1.0)


def invert_marmit(observed: np.ndarray, theta: float = 0.5, eps: float = 1e-6) -> np.ndarray:
    """Invert placeholder MARMIT transform to pseudo moisture proxy."""
    return observed / (theta + eps)
