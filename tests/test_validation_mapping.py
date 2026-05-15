import numpy as np

from soil_moisture_hyperspectral.mapping import create_moisture_map, export_geotiff_placeholder
from soil_moisture_hyperspectral.validation import mae, rmse


def test_metrics_zero_error_case():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert rmse(y_true, y_pred) == 0.0
    assert mae(y_true, y_pred) == 0.0


def test_create_moisture_map_clips_bounds():
    pred = np.array([[-0.2, 0.4, 1.4]])
    out = create_moisture_map(pred)
    assert np.min(out) >= 0.0
    assert np.max(out) <= 1.0


def test_export_geotiff_placeholder_metadata():
    arr = np.zeros((3, 4), dtype=np.float32)
    meta = export_geotiff_placeholder(arr, "output.tif")
    assert meta["shape"] == (3, 4)
    assert meta["written"] is False
