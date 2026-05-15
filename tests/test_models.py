import numpy as np

from soil_moisture_hyperspectral.calibration import fit_linear_calibration, split_train_test_by_index
from soil_moisture_hyperspectral.marmit import invert_marmit, marmit_forward_model
from soil_moisture_hyperspectral.optram import estimate_optram_moisture


def test_optram_output_shape():
    ndvi = np.array([[0.2, 0.4]])
    str_index = np.array([[0.3, 0.5]])
    out = estimate_optram_moisture(ndvi, str_index)
    assert out.shape == ndvi.shape


def test_marmit_forward_inverse_roundtrip_like_behavior():
    refl = np.array([0.2, 0.4, 0.6])
    modeled = marmit_forward_model(refl, theta=0.5)
    inv = invert_marmit(modeled, theta=0.5)
    assert inv.shape == refl.shape


def test_split_train_test_sizes():
    train_idx, test_idx = split_train_test_by_index(10, test_fraction=0.2)
    assert len(train_idx) == 8
    assert len(test_idx) == 2


def test_fit_linear_calibration_returns_scalars():
    x = np.array([0.1, 0.2, 0.3, 0.4])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    a, b = fit_linear_calibration(x, y)
    assert isinstance(a, float)
    assert isinstance(b, float)
