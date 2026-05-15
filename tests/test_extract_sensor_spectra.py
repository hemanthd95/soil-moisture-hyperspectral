from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.extract_sensor_spectra import extract_sensor_spectra


@pytest.mark.skipif(pytest.importorskip("rasterio") is None, reason="rasterio unavailable")
def test_extract_sensor_spectra_with_multiple_radii(tmp_path: Path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    raster_path = tmp_path / "cube.bip"
    data = np.zeros((3, 10, 10), dtype=np.float32)
    data[0] = 0.2
    data[1] = 0.4
    data[2] = 0.6
    data[:, 5, 5] = [0.3, 0.5, 0.7]

    with rasterio.open(
        raster_path,
        "w",
        driver="ENVI",
        height=10,
        width=10,
        count=3,
        dtype="float32",
        transform=from_origin(500000.0, 3600000.0, 1.0, 1.0),
        crs="EPSG:32617",
        nodata=-9999.0,
    ) as dst:
        dst.write(data)

    sensors = pd.DataFrame(
        {
            "sensor_id": ["S1"],
            "easting": [500005.5],
            "northing": [3599994.5],
        }
    )
    wavelengths = [500.0, 600.0, 700.0]

    out = extract_sensor_spectra(
        raster_path=raster_path,
        wavelengths=wavelengths,
        sensors=sensors,
        buffer_radii_m=[1.0, 2.0],
        flight_id="F1",
        poor_valid_threshold=100,
    )

    assert len(out) == 2
    assert set(out["buffer_radius_m"].tolist()) == {1.0, 2.0}
    assert all(col in out.columns for col in ["wl_500_mean", "wl_500_median", "wl_500_std"])
    assert out.iloc[0]["sensor_id"] == "S1"
    assert out.iloc[0]["flight_id"] == "F1"
    assert out.iloc[0]["valid_pixel_count"] > 0
    assert bool(out.iloc[0]["poor_valid"]) is True


@pytest.mark.skipif(pytest.importorskip("rasterio") is None, reason="rasterio unavailable")
def test_extract_sensor_spectra_with_masks(tmp_path: Path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    raster_path = tmp_path / "cube2.bip"
    data = np.ones((2, 6, 6), dtype=np.float32)

    with rasterio.open(
        raster_path,
        "w",
        driver="ENVI",
        height=6,
        width=6,
        count=2,
        dtype="float32",
        transform=from_origin(100.0, 200.0, 1.0, 1.0),
        crs="EPSG:32617",
    ) as dst:
        dst.write(data)

    sensors = pd.DataFrame({"sensor_id": ["S2"], "easting": [103.5], "northing": [196.5]})
    vegetation_mask = np.zeros((6, 6), dtype=bool)
    vegetation_mask[2:5, 2:5] = True

    out = extract_sensor_spectra(
        raster_path=raster_path,
        wavelengths=[500.0, 600.0],
        sensors=sensors,
        buffer_radii_m=1.0,
        vegetation_mask=vegetation_mask,
    )

    assert len(out) == 1
    assert int(out.iloc[0]["valid_pixel_count"]) < 9
