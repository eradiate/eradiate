import numpy as np
import pytest

from eradiate.quad import Quad
from eradiate.radprops import absdb_factory
from eradiate.spectral import CKDSpectralIndex
from eradiate.spectral.ckd_quad import CKDQuadConfig
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


def test_mono_spectral_grid_default():
    grid = MonoSpectralGrid.default()
    np.testing.assert_allclose(grid.wavelengths.m, np.arange(250.0, 3126.0, 1.0))


def test_mono_spectral_grid_from_absorption_database():
    abs_db = absdb_factory.create("komodo")
    grid = MonoSpectralGrid.from_absorption_database(abs_db)
    np.testing.assert_allclose(
        grid.wavelengths.m, abs_db.spectral_coverage.index.get_level_values(1).values
    )


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


def test_mono_spectral_grid_merge():
    grid_1 = MonoSpectralGrid(wavelengths=np.arange(500.0, 601.0, 10.0))
    grid_2 = MonoSpectralGrid(wavelengths=np.arange(550.0, 651.0, 10.0))
    grid_3 = grid_1.merge(grid_2)
    np.testing.assert_array_equal(grid_3.wavelengths.m, np.arange(500.0, 651.0, 10.0))


def test_mono_spectral_grid_walk_indices():
    grid = MonoSpectralGrid([300, 400, 500])
    np.testing.assert_allclose(
        [x.w.m_as("nm") for x in grid.walk_indices()], grid.wavelengths.m
    )


def test_ckd_spectral_grid_construct():
    # Regular constructor
    grid = CKDSpectralGrid([495, 505], [505, 515])
    np.testing.assert_array_equal(grid.wcenters.m_as("nm"), [500, 510])

    # Range constructor
    grid = CKDSpectralGrid.arange(500, 520, 10)
    np.testing.assert_array_equal(grid.wcenters.m_as("nm"), [500, 510])


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


def test_ckd_spectral_grid_default():
    grid = CKDSpectralGrid.default()
    expected_wcenters = np.arange(250.0, 3126.0, 10.0)
    np.testing.assert_allclose(grid.wcenters.m, expected_wcenters)
    np.testing.assert_allclose(grid.wmins.m, expected_wcenters - 5.0)
    np.testing.assert_allclose(grid.wmaxs.m, expected_wcenters + 5.0)


def test_ckd_spectral_grid_arange():
    grid = CKDSpectralGrid.arange(540.0, 570.0, 10.0)
    np.testing.assert_allclose(grid.wmins.m, [535.0, 545.0, 555.0])
    np.testing.assert_allclose(grid.wmaxs.m, [545.0, 555.0, 565.0])
    np.testing.assert_allclose(grid.wcenters.m, [540.0, 550.0, 560.0])


def test_ckd_spectral_grid_from_absorption_database(mode_ckd):
    # The 'monotropa' database is a notable problematic case: its wavelength
    # coordinate is set to values that match the central wavenumber of each bin,
    # and not to the middle of the spectral interval in the wavelength space.
    abs_db = absdb_factory.create("monotropa")
    grid = CKDSpectralGrid.from_absorption_database(abs_db)
    wcenters_expected = abs_db.spectral_coverage.index.get_level_values(1).values
    np.testing.assert_allclose(grid.wcenters.m, wcenters_expected)


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
    grid = CKDSpectralGrid.arange(start=500.0, stop=610.0, step=10.0)
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
    grid = CKDSpectralGrid.arange(start=500.0, stop=610.0, step=10.0)
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


def test_ckd_spectral_grid_merge():
    # Merge two grids with an overlap
    grid_1 = CKDSpectralGrid.arange(500.0, 601.0, 10.0)
    grid_2 = CKDSpectralGrid.arange(550.0, 651.0, 10.0)
    grid_3 = grid_1.merge(grid_2)
    np.testing.assert_array_equal(grid_3.wmins.m, np.arange(495.0, 646.0, 10.0))
    np.testing.assert_array_equal(grid_3.wmaxs.m, np.arange(505.0, 656.0, 10.0))
    np.testing.assert_array_equal(grid_3.wcenters.m, np.arange(500.0, 651.0, 10.0))

    # Merge two disjoint grids
    grid_1 = CKDSpectralGrid.arange(550.0, 561.0, 10.0)
    grid_2 = CKDSpectralGrid.arange(500.0, 511.0, 10.0)
    grid_3 = grid_1.merge(grid_2)
    np.testing.assert_array_equal(grid_3.wmins.m, [495.0, 505.0, 545.0, 555.0])
    np.testing.assert_array_equal(grid_3.wmaxs.m, [505.0, 515.0, 555.0, 565.0])
    np.testing.assert_array_equal(grid_3.wcenters.m, [500.0, 510.0, 550.0, 560.0])


def test_ckd_grid_walk_indices():
    grid = CKDSpectralGrid.arange(540, 560, 10)
    cqc = CKDQuadConfig(type="gauss_legendre", ng_max=4, policy="fixed")
    expected_gs = Quad.gauss_legendre(4).eval_nodes([0, 1])

    expected_sequence = [
        CKDSpectralIndex(w, g) for w in grid.wcenters for g in expected_gs
    ]
    actual_sequence = list(grid.walk_indices(cqc))

    np.testing.assert_allclose(
        [x.w.m_as("nm") for x in actual_sequence],
        [x.w.m_as("nm") for x in expected_sequence],
    )
    np.testing.assert_allclose(
        [x.g for x in actual_sequence], [x.g for x in expected_sequence]
    )
