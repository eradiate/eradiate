import numpy as np
import pytest
import xarray as xr

from eradiate import unit_registry as ureg
from eradiate.radprops.mie import (
    make_data_set,
    compute_mono_properties_single_radius,
    compute_mono_properties_multiple_radius,
    compute_mono_properties,
    compute_properties,
)


def test_make_data_test():
    """
    Returns a xarray.Dataset.
    """
    ds = make_data_set(
        phase=np.random.rand((50)),
        xs_t=1e-2,
        albedo=0.96,
        w=550.0,
        r=1.0,
        m=1.5 - 0.01j,
        mu=np.linspace(-1, 1),
    )
    assert isinstance(ds, xr.Dataset)


def test_compute_mono_properties_single_radius():
    """
    Returns a xarray.Dataset.
    """
    ds = compute_mono_properties_single_radius(
        w=550.0,
        r=1.0,
        m=1.5 - 0.1j,
    )
    assert isinstance(ds, xr.Dataset)


def test_compute_mono_properties_multiple_radius():
    """
    Returns a xarray.Dataset.
    """
    ds = compute_mono_properties_multiple_radius(
        w=550.0,
        r=np.linspace(1.0, 2.0),
        m=1.5 - 0.1j,
    )
    assert isinstance(ds, xr.Dataset)


def lognorm(r0: float, std: float = 0.4):
    """
    Return a log-normal distribution
    """
    return lambda r: (1.0 / (np.log(10) * r * std * np.sqrt(2 * np.pi))) * np.exp(
        -np.square(np.log10(r) - np.log10(r0)) / (2 * np.square(std))
    )


def make_data_array(r: np.ndarray, rw: np.ndarray) -> xr.DataArray:
    return xr.DataArray(
        data=rw,
        coords=[("r", r)],
        attrs=dict(
            units="",
        ),
    )


def test_compute_mono_properties():
    """
    Returns a xarray.Dataset.
    """
    distribution = lognorm(r0=1.0, std=0.4)
    r = np.geomspace(0.1, 5.0)
    rw = distribution(r=r)
    rdist = make_data_array(r=r, rw=rw)

    ds = compute_mono_properties(
        w=550.0,
        rdist=rdist,
        m=1.5 - 0.1j,
    )
    assert isinstance(ds, xr.Dataset)


@pytest.fixture
def test_refractive_index():
    return xr.DataArray(
        np.array([1.5 - 0.1j, 1.5 - 0.01j], dtype=complex),
        dims="w",
        coords={
            "w": ("w", np.array([240.0, 2800.0]), dict(units="nm")),
        },
    )


def test_compute_properties(test_refractive_index):
    """
    Returns a xarray.Dataset.
    """
    distribution = lognorm(r0=1.0, std=0.4)
    r = np.geomspace(0.1, 5.0)
    rw = distribution(r=r)
    rdist = make_data_array(r=r, rw=rw)

    ds = compute_properties(
        w=np.linspace(500, 600, 3),
        rdist=rdist,
        m=test_refractive_index,
    )
    assert isinstance(ds, xr.Dataset)
