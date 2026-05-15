# Project Brief: Hyperspectral Soil Moisture Modeling

Goal:
Develop a Python-based modeling pipeline to predict near-surface soil moisture from UAV hyperspectral imagery using OPTRAM, MARMIT/MARMIT-2, and hybrid calibration with ground-truth soil moisture sensors.

Study site:
Olar Research and Demonstration Farm, South Carolina.

Data:
- UAV hyperspectral imagery from Resonon Pika L, approximately 400–1000 nm.
- UAV hyperspectral imagery from Pika IR-L, approximately 900–1700 nm.
- Ground-truth soil moisture from 31–36 sensors at 5 cm depth.
- Sensor coordinates in EPSG:32617.
- Soil types: UcB, BaB, NrA.
- Multiple flight dates under different soil moisture conditions.
- Reflectance calibrated imagery.
- Georeferenced imagery using QGIS/Spectronon workflow.

Modeling objectives:
1. Extract reflectance spectra around each sensor.
2. Remove non-soil, vegetation, white paint, shadow, and noisy pixels.
3. Implement OPTRAM using NDVI and STR.
4. Implement MARMIT/MARMIT-2 style radiative transfer calibration.
5. Compare OPTRAM, MARMIT, spectral indices, and ML baselines.
6. Validate using spatial, temporal, and soil-type holdouts.
7. Export soil moisture maps as GeoTIFFs.
8. Produce figures and tables for manuscript/proposal use.

Constraints:
- Data are large; code must use chunked processing with rasterio, dask, xarray, or zarr where appropriate.
- Do not load entire hyperspectral cubes into memory unless explicitly requested.
- All outputs must be reproducible from config files.
