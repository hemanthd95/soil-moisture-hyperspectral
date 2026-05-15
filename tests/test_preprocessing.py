import numpy as np

from soil_moisture_hyperspectral.preprocessing import compute_ndvi, compute_str, mask_invalid_pixels


def test_mask_invalid_pixels_replaces_nan():
    cube = np.array([[[1.0, np.nan], [2.0, 3.0]]])
    out = mask_invalid_pixels(cube, invalid_value=-9999.0)
    assert out[0, 0, 1] == -9999.0


def test_compute_ndvi_shape():
    nir = np.array([[0.6, 0.5]])
    red = np.array([[0.2, 0.1]])
    ndvi = compute_ndvi(nir, red)
    assert ndvi.shape == nir.shape


def test_compute_str_non_negative_for_fractional_reflectance():
    swir = np.array([[0.2, 0.4, 0.8]])
    out = compute_str(swir)
    assert np.all(out >= 0)
