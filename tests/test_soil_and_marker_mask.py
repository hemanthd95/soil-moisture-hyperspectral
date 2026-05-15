import numpy as np

from src.preprocessing.marker_mask import white_marker_mask
from src.preprocessing.soil_mask import compute_ndvi_from_cube, shadow_mask, vegetation_mask


def test_vegetation_and_shadow_masks_shapes():
    wavelengths = [500.0, 670.0, 800.0]
    cube = np.zeros((3, 4, 4), dtype=float)
    cube[1] = 0.2  # red
    cube[2] = 0.6  # nir

    cfg = {"thresholds": {"ndvi_veg_min": 0.4, "dark_reflectance_max": 0.1}}
    ndvi = compute_ndvi_from_cube(cube, wavelengths)
    veg = vegetation_mask(cube, wavelengths, cfg)
    dark = shadow_mask(cube, wavelengths, cfg)

    assert ndvi.shape == (4, 4)
    assert veg.shape == (4, 4)
    assert dark.shape == (4, 4)
    assert np.all(veg)
    assert not np.any(dark)


def test_white_marker_mask_shape_and_detection():
    wavelengths = [450.0, 500.0, 550.0, 600.0, 650.0, 700.0]
    cube = np.full((6, 3, 3), 0.2, dtype=float)
    cube[:, 1, 1] = [0.95, 0.75, 0.95, 0.75, 0.95, 0.75]

    cfg = {"thresholds": {"white_visible_min": 0.7, "white_roughness_min": 0.1}}
    mask = white_marker_mask(cube, wavelengths, cfg)

    assert mask.shape == (3, 3)
    assert bool(mask[1, 1]) is True
