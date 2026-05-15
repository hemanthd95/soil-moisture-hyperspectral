# Soil Moisture Hyperspectral Modeling (Skeleton)

This repository contains a clean Python project skeleton for near-surface soil moisture modeling from UAV hyperspectral imagery. It is aligned with `PROJECT_BRIEF.md` and intentionally includes placeholder logic only.

## Workflow Overview

1. **IO** (`io.py`)
   - Path handling and lightweight data access scaffolding.
   - Placeholder cube-opening metadata without loading full arrays.

2. **Preprocessing** (`preprocessing.py`)
   - Pixel masking for invalid values.
   - Basic spectral index placeholders (NDVI and STR).

3. **OPTRAM** (`optram.py`)
   - Placeholder OPTRAM moisture estimation from NDVI/STR and dry/wet edge lines.

4. **MARMIT** (`marmit.py`)
   - Placeholder forward/inverse transforms to scaffold MARMIT/MARMIT-2 integration.

5. **Calibration** (`calibration.py`)
   - Deterministic split helpers and simple linear fit placeholders.

6. **Validation** (`validation.py`)
   - Basic metrics (RMSE, MAE).

7. **Mapping** (`mapping.py`)
   - Moisture clipping and placeholder GeoTIFF-export metadata.

## Project Structure

```text
src/soil_moisture_hyperspectral/
  __init__.py
  io.py
  preprocessing.py
  optram.py
  marmit.py
  calibration.py
  validation.py
  mapping.py
tests/
  test_preprocessing.py
  test_models.py
  test_validation_mapping.py
environment.yml
README.md
```

## Environment Setup (Ubuntu)

```bash
conda env create -f environment.yml
conda activate soil-moisture-hyperspectral
```

## Testing

```bash
pytest -q
```

## Notes

- This skeleton does **not** assume any real hyperspectral or sensor files are present.
- No large raw data files are included or referenced.
- Future work should add config-driven pipelines and chunked raster processing for large cubes.
