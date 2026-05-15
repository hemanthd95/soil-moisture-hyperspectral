from pathlib import Path

import numpy as np
import pytest

from src.io.read_bip import read_bip_metadata, read_selected_bands, read_window


@pytest.mark.skipif(pytest.importorskip("rasterio") is None, reason="rasterio unavailable")
def test_read_bip_metadata_and_subsets(tmp_path: Path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    image_path = tmp_path / "scene.bip"
    hdr_path = tmp_path / "scene.hdr"

    data = np.arange(3 * 5 * 6, dtype=np.float32).reshape(3, 5, 6)
    with rasterio.open(
        image_path,
        "w",
        driver="ENVI",
        height=5,
        width=6,
        count=3,
        dtype="float32",
        transform=from_origin(500000.0, 3600000.0, 1.0, 1.0),
        crs="EPSG:32617",
    ) as dst:
        dst.write(data)

    hdr_path.write_text(
        "ENVI\n"
        "samples = 6\n"
        "lines = 5\n"
        "bands = 3\n"
        "interleave = bip\n"
        "data type = 4\n"
        "wavelength = {500, 600, 700}\n"
        "coordinate system string = PROJCS[\"WGS 84 / UTM zone 17N\"]\n"
        "map info = {UTM, 1.0, 1.0, 500000, 3600000, 1, 1, 17, North, WGS-84}\n",
        encoding="utf-8",
    )

    meta = read_bip_metadata(image_path)
    assert meta["band_count"] == 3
    assert meta["height"] == 5
    assert meta["width"] == 6
    assert meta["wavelengths"] == [500.0, 600.0, 700.0]

    subset = read_selected_bands(image_path, [1, 3])
    assert subset.shape == (2, 5, 6)
    assert np.isclose(subset[0, 0, 0], data[0, 0, 0])

    win = read_window(image_path, row=2, col=2, half_size=1, band_indices=[2])
    assert win.shape == (1, 3, 3)


@pytest.mark.skipif(pytest.importorskip("rasterio") is None, reason="rasterio unavailable")
def test_read_window_by_map_coordinate(tmp_path: Path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    image_path = tmp_path / "scene2.bip"
    data = np.ones((2, 4, 4), dtype=np.float32)

    with rasterio.open(
        image_path,
        "w",
        driver="ENVI",
        height=4,
        width=4,
        count=2,
        dtype="float32",
        transform=from_origin(100.0, 200.0, 10.0, 10.0),
        crs="EPSG:32617",
    ) as dst:
        dst.write(data)

    win = read_window(image_path, x=115.0, y=185.0, half_size=1)
    assert win.shape[1:] == (3, 3)
