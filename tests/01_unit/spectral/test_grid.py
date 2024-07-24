import numpy as np
import pytest

from eradiate.spectral.grid import CKDSpectralGrid, MonoSpectralGrid
from eradiate.spectral.response import BandSRF, DeltaSRF, UniformSRF
from eradiate.units import unit_registry as ureg


@pytest.mark.parametrize(
    "wavelengths, expected",
    [
        (np.arange(500, 601, 10), np.arange(500.0, 601.0, 10.0)),
        (550, [550.0]),
        ([500, 400], [400.0, 500.0]),
        ([600, 500, 600], [500.0, 600.0]),
    ],
    ids=[
        "vector",  # The regular constructor
        "scalar",  # Input a single wavelength as a scalar
        "order",  # Input wavelengths unordered, constructor should order them
        "uniqueness",  # Input duplicate wavelengths, only 1 occurrence is kept
    ],
)
def test_mono_spectral_grid_construct(wavelengths, expected):
    grid = MonoSpectralGrid(wavelengths=wavelengths)
    assert grid.wavelengths.m.dtype == np.dtype("float64")
    np.testing.assert_allclose(grid.wavelengths.m_as("nm"), expected)


def test_mono_spectral_grid_select_delta_srf():
    # SRF wavelengths are just passed through
    grid = MonoSpectralGrid(wavelengths=np.arange(500.0, 601.0, 10.0))
    srf = DeltaSRF(wavelengths=[525, 550, 575])
    np.testing.assert_allclose(grid.select(srf).wavelengths.m, [525, 550, 575])


@pytest.mark.parametrize(
    "wmin, wmax, expected",
    [
        (530, 570, [530, 540, 550, 560, 570]),
    ],
    ids=[
        "basic",  # Wavelength values matching bounds are selected
    ],
)
def test_mono_spectral_grid_select_uniform_srf(wmin, wmax, expected):
    grid = MonoSpectralGrid(wavelengths=np.arange(500.0, 601.0, 10.0))
    srf = UniformSRF(wmin=wmin, wmax=wmax)
    np.testing.assert_allclose(grid.select(srf).wavelengths.m, expected)


@pytest.mark.parametrize(
    "band_srf_kwargs, expected_selected_w",
    [
        (
            {
                "wavelengths": np.linspace(500.0, 600.0, 11),
                "values": ([0] + [1] * 9 + [0]) * ureg.dimensionless,
            },
            np.arange(510, 591, 10) * ureg.nm,
        ),
        (
            {
                "wavelengths": np.linspace(500.0, 600.0, 11),
                "values": ([0, 0] + [1] * 7 + [0, 0]) * ureg.dimensionless,
            },
            np.arange(520, 581, 10) * ureg.nm,
        ),
        (
            {
                "wavelengths": np.linspace(505.0, 605.0, 11),
                "values": ([0, 0] + [1] * 7 + [0, 0]) * ureg.dimensionless,
            },
            [520.0, 530.0, 540.0, 550.0, 560.0, 570.0, 580.0, 590.0] * ureg.nm,
        ),
        (
            {
                "wavelengths": np.linspace(500.0, 600.0, 11),
                "values": [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0] * ureg.dimensionless,
            },
            [520, 570, 580] * ureg.nm,
        ),
    ],
    ids=[
        "connected_1",
        "connected_2",
        "connected_3",
        "nonconnected",
    ],
)
def test_mono_spectral_grid_select_band_srf(band_srf_kwargs, expected_selected_w):
    grid = MonoSpectralGrid(wavelengths=np.arange(500.0, 601.0, 10.0))
    srf = BandSRF(**band_srf_kwargs)
    grid_selected = grid.select(srf)
    np.testing.assert_allclose(grid_selected.wavelengths.m, expected_selected_w.m)


def test_ckd_spectral_grid_construct():
    # Regular constructor
    grid = CKDSpectralGrid([495, 505], [505, 515])
    np.testing.assert_array_equal(grid._wcenters.m_as("nm"), [500, 510])

    # Range constructor
    grid = CKDSpectralGrid.arange(500, 510, 10)
    np.testing.assert_array_equal(grid._wcenters.m_as("nm"), [500, 510])


@pytest.mark.parametrize(
    "policy, expected",
    [
        ("raise", ValueError),
        ("keep_min", {"min": [495, 505, 515], "max": [505, 515, 525]}),
        ("keep_max", {"min": [495, 505, 515 + 1e-6], "max": [505, 515 + 1e-6, 525]}),
    ],
)
def test_ckd_spectral_grid_construct_fix_mismatch(policy, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            CKDSpectralGrid(
                [495, 505, 515],
                [505, 515 + 1e-6, 525],
                fix_bounds=policy,
                epsilon=1e-5,
            )

    else:
        grid = CKDSpectralGrid(
            [495, 505, 515],
            [505, 515 + 1e-6, 525],
            fix_bounds=policy,
            epsilon=1e-5,
        )
        np.testing.assert_array_equal(grid.wmins.m_as("nm"), expected["min"])
        np.testing.assert_array_equal(grid.wmaxs.m_as("nm"), expected["max"])


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
        "multiple_values",
        "many_values",  # If more values than bins are passed, no duplicate selection
        "priority_left",  # When wavelength falls between two bins, left bin is selected by convention
    ],
)
def test_ckd_grid_select_delta_srf(delta_wavelengths, expected_selected_wcenters):
    grid = CKDSpectralGrid.arange(start=500.0, stop=600.0, step=10.0)
    srf = DeltaSRF(wavelengths=delta_wavelengths)
    selected = grid.select(srf)
    np.testing.assert_allclose(selected.wcenters.m, expected_selected_wcenters.m)


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
    np.testing.assert_allclose(selected.wcenters.m, expected_selected_wcenters.m)


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
                "wavelengths": np.linspace(505.0, 605.0, 11),
                "values": ([0, 0] + [1] * 7 + [0, 0]) * ureg.dimensionless,
            },
            [520.0, 530.0, 540.0, 550.0, 560.0, 570.0, 580.0, 590.0] * ureg.nm,
        ),
        (
            {
                "wavelengths": np.linspace(500.0, 600.0, 11),
                "values": [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0] * ureg.dimensionless,
            },
            [510, 520, 530, 560, 570, 580, 590] * ureg.nm,
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
    np.testing.assert_allclose(grid_selected.wcenters.m, expected_selected_wcenters.m)
