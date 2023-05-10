"""Test cases for the srf_filter module."""
import datetime

import numpy as np
import pytest
import xarray as xr

from eradiate.srf_tools import integral_filter, spectral_filter, threshold_filter, trim
from eradiate.units import unit_registry as ureg

SRF_VALUES_RELEVANT = np.array([0.0, 0.5, 1.0, 0.5, 0.0])


@pytest.fixture
def srf_values_trailing_and_leading_zeros() -> np.ndarray:
    return np.concatenate(
        [
            np.zeros(50),
            SRF_VALUES_RELEVANT,
            np.zeros(200),
        ]
    )


@pytest.fixture
def srf_values_trailing_zeros() -> np.ndarray:
    return np.concatenate(
        [
            SRF_VALUES_RELEVANT,
            np.zeros(200),
        ]
    )


@pytest.fixture
def srf_values_leading_zeros() -> np.ndarray:
    return np.concatenate(
        [
            np.zeros(50),
            SRF_VALUES_RELEVANT,
        ]
    )


@pytest.mark.parametrize(
    "srf_values",
    [
        "srf_values_trailing_and_leading_zeros",
        "srf_values_trailing_zeros",
        "srf_values_leading_zeros",
    ],
)
def test_trim(request, srf_values) -> None:
    """Trim all leading zeros except last and all trailing zeros except first."""
    srf_values = request.getfixturevalue(srf_values)
    utcnow = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    ds = xr.Dataset(
        {"srf": ("w", srf_values, {"units": "dimensionless"})},
        coords={"w": ("w", np.linspace(300, 800, srf_values.size), {"units": "nm"})},
        attrs={"history": f"{utcnow} - data set creation"},
    )
    trimmed = trim(srf=ds)
    assert trimmed.srf.size == SRF_VALUES_RELEVANT.size


def test_threshold_filter() -> None:
    """
    Drop data points where response is smaller or equal than a threshold value.
    """
    srf_values = np.array([1e-6, 1e-3, 1e-2, 1e0, 1e-1, 1e-2, 1e-2, 1e-5])
    utcnow = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    srf = xr.Dataset(
        {"srf": ("w", srf_values, {"units": "dimensionless"})},
        coords={"w": ("w", np.linspace(300, 800, srf_values.size), {"units": "nm"})},
        attrs={"history": f"{utcnow} - data set creation"},
    )
    threshold = 0.002
    filtered = threshold_filter(srf=srf, value=threshold)

    assert srf_values[srf_values > threshold].size == filtered.srf.size


@pytest.mark.parametrize(
    "wrange",
    [
        (500.0 * ureg.nm, None),
        (None, 600.0 * ureg.nm),
        (500.0 * ureg.nm, 600.0 * ureg.nm),
        (None, None),
    ],
)
def test_spectral_filter(wrange) -> None:
    """
    Drop points falling out of wavelength range specified by wmin and wmax.
    """
    wmin, wmax = wrange

    w = np.linspace(400, 800)
    srf_values = np.random.random(w.size)
    utcnow = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    srf = xr.Dataset(
        {"srf": ("w", srf_values, {"units": "dimensionless"})},
        coords={"w": ("w", w, {"units": "nm"})},
        attrs={"history": f"{utcnow} - data set creation"},
    )
    filtered = spectral_filter(srf=srf, wmin=wmin, wmax=wmax)

    if wmax is None and wmin is not None:
        assert srf_values[w >= wmin.m].size == filtered.srf.size
    if wmin is None and wmax is not None:
        assert srf_values[w <= wmax.m].size == filtered.srf.size
    if wmin is not None and wmax is not None:
        assert srf_values[(w <= wmax.m) & (w >= wmin.m)].size == filtered.srf.size
    if wmin is None and wmax is None:
        assert srf_values.size == filtered.srf.size


@pytest.mark.parametrize(
    "percentage",
    [50.0, 90.0, 95.0, 99.0],
)
def test_integral_filter(percentage) -> None:
    """
    Keep only data that contribute to the integrated spectral response value
    to the amount of the specified percentage.
    """

    srf_values = np.ones(100000)
    w_values = np.linspace(300, 800, srf_values.size)

    utcnow = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    srf = xr.Dataset(
        {"srf": ("w", srf_values, {"units": "dimensionless"})},
        coords={"w": ("w", w_values, {"units": "nm"})},
        attrs={"history": f"{utcnow} - data set creation"},
    )
    filtered = integral_filter(srf=srf, percentage=percentage)

    assert int(100 * filtered.srf.size / srf_values.size) == percentage
