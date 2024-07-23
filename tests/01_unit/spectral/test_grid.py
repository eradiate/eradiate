import numpy as np
import pytest

from eradiate.spectral.grid import CKDSpectralGrid
from eradiate.spectral.response import BandSRF, DeltaSRF, UniformSRF
from eradiate.units import unit_registry as ureg


def test_ckd_spectral_grid_construct():
    # Regular constructor
    grid = CKDSpectralGrid([495, 505], [505, 515])
    np.testing.assert_array_equal(grid.wcenters.m_as("nm"), [500, 510])

    # Range constructor
    grid = CKDSpectralGrid.arange(500, 510, 10)
    np.testing.assert_array_equal(grid.wcenters.m_as("nm"), [500, 510])


@pytest.mark.parametrize(
    "delta_wavelengths, expected_selected_wcenters",
    [
        ([550.0] * ureg.nm, [550.0] * ureg.nm),
        ([500.0, 600.0] * ureg.nm, [500.0, 600.0] * ureg.nm),
        (
            np.linspace(500.0, 600.0, 100) * ureg.nm,
            np.arange(500.0, 601.0, 10.0) * ureg.nm,
        ),
        ([505.0] * ureg.nm, [500.0] * ureg.nm),
    ],
    ids=[
        "single_value",
        "multiple_value",
        # If more values than bins are passed, still works as expected
        "many_values",
        # When wavelength falls between two bins, left bin is selected by convention
        "priority_left",
    ],
)
def test_ckd_grid_select_delta_srf(delta_wavelengths, expected_selected_wcenters):
    grid = CKDSpectralGrid.arange(start=500.0, stop=600.0, step=10.0)
    srf = DeltaSRF(wavelengths=delta_wavelengths)
    selected = grid.select(srf)
    assert np.allclose(selected.wcenters, expected_selected_wcenters)


@pytest.mark.parametrize(
    "uniform_bounds, expected_selected_wcenters",
    [
        ([500, 600] * ureg.nm, np.arange(500, 601, 10) * ureg.nm),
        ([505, 595] * ureg.nm, np.arange(510, 591, 10) * ureg.nm),
    ],
    ids=[
        "basic",
        "bounds_on_nodes",
    ],
)
def test_ckd_grid_select_uniform_srf(uniform_bounds, expected_selected_wcenters):
    grid = CKDSpectralGrid.arange(start=500.0, stop=600.0, step=10.0)
    srf = UniformSRF(*uniform_bounds, 1.0)
    selected = grid.select(srf)
    assert np.allclose(selected.wcenters, expected_selected_wcenters)


@pytest.mark.parametrize(
    "band_srf_kwargs, expected_selected_wcenters",
    [
        (
            {
                "wavelengths": np.linspace(500.0, 600.0, 11),
                "values": ([0] + [1] * 9 + [0]) * ureg.dimensionless,
            },
            np.arange(500, 601, 10) * ureg.nm,
        ),
        (
            {
                "wavelengths": np.linspace(500.0, 600.0, 11),
                "values": ([0, 0] + [1] * 7 + [0, 0]) * ureg.dimensionless,
            },
            np.arange(510, 591, 10) * ureg.nm,
        ),
        (
            {
                "wavelengths": np.linspace(500.0, 600.0, 11),
                "values": [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0] * ureg.dimensionless,
            },
            [510, 520, 530, 560, 570, 580, 590] * ureg.nm,
        ),
        (
            {
                "wavelengths": np.linspace(505.0, 605.0, 11),
                "values": ([0, 0] + [1] * 7 + [0, 0]) * ureg.dimensionless,
            },
            [520.0, 530.0, 540.0, 550.0, 560.0, 570.0, 580.0, 590.0] * ureg.nm,
        ),
    ],
    ids=[
        "connected_1",
        "connected_2",
        "connected_3",
        "nonconnected",
    ],
)
def test_ckd_grid_select_band_srf(band_srf_kwargs, expected_selected_wcenters):
    grid = CKDSpectralGrid.arange(start=280.0, stop=2400.0, step=10.0)
    srf = BandSRF(**band_srf_kwargs)
    grid_selected = grid.select(srf)
    assert np.allclose(grid_selected.wcenters, expected_selected_wcenters)
