"""OPTRAM model scaffolding."""

import numpy as np


def estimate_optram_moisture(
    ndvi: np.ndarray,
    str_index: np.ndarray,
    wet_edge: tuple[float, float] = (0.0, 1.0),
    dry_edge: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Estimate soil moisture using a placeholder OPTRAM relationship.

    Parameters
    ----------
    ndvi
        NDVI image.
    str_index
        STR image.
    wet_edge, dry_edge
        Placeholder line parameters for wet and dry edges.
    """
    wet = wet_edge[0] * ndvi + wet_edge[1]
    dry = dry_edge[0] * ndvi + dry_edge[1]
    denom = np.where((wet - dry) == 0.0, np.nan, wet - dry)
    return (str_index - dry) / denom
