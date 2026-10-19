import numpy as np
import pytest

from eradiate import unit_registry as ureg
from eradiate.contexts import KernelContext
from eradiate.grid import SphericalGrid
from eradiate.scenes._gridvolume import make_volume_grid
from eradiate.scenes.geometry import SphericalShellGeometry


def _spherical_grid(azimuths_deg):
    return SphericalGrid(
        levels=np.array([0.0, 1.0]) * ureg.km,
        azimuths=np.asarray(azimuths_deg) * ureg.deg,
        colatitudes=np.array([0.0, 180.0]) * ureg.deg,
    )


@pytest.mark.parametrize(
    "azimuths, expected",
    [
        ([-180.0, 0.0, 180.0], (-180.0, 180.0, 0)),
        ([200.0, 225.0, 250.0], (-160.0, -110.0, 0)),
        ([0.0, 360.0], (-180.0, 180.0, 0)),
        (np.linspace(0.0, 360.0, 5), (-180.0, 180.0, 2)),
        (np.linspace(-90.0, 270.0, 5), (-180.0, 180.0, 3)),
    ],
    ids=["native", "shifted", "full_single_cell", "full_0_360", "full_m90_270"],
)
def test_spherical_grid_kernel_azimuths(azimuths, expected):
    phi_min, phi_max, n_roll = _spherical_grid(azimuths).kernel_azimuths()
    np.testing.assert_allclose(np.rad2deg([phi_min, phi_max]), expected[:2])
    assert n_roll == expected[2]


@pytest.mark.parametrize(
    "azimuths",
    [[170.0, 190.0], np.linspace(0.0, 360.0, 4), [0.0, 400.0], [10.0, 0.0]],
    ids=["partial_across_cut", "cut_inside_cell", "too_wide", "decreasing"],
)
def test_spherical_grid_kernel_azimuths_invalid(azimuths):
    with pytest.raises(ValueError):
        _spherical_grid(azimuths)


def test_spherical_grid_to_world(mode_mono):
    """to_world maps the azimuth range to its place in [0, 1]."""
    to_world = _spherical_grid([200.0, 225.0, 250.0]).to_world
    z = [(to_world @ [0.0, 0.0, w])[2] for w in (0.0, 1.0)]
    np.testing.assert_allclose(z, np.array([-160.0, -110.0]) / 360.0 + 0.5)


def test_make_volume_grid_spherical_roll(mode_mono):
    """Volume data on [0°, 360°] is rolled to start at the -180° cell."""
    grid = _spherical_grid(np.linspace(0.0, 360.0, 5))
    geometry = SphericalShellGeometry(grid=grid)
    sectors = grid.sectors.m_as(ureg.deg)  # [45, 135, 225, 315]

    result = make_volume_grid(
        geometry,
        KernelContext(),
        eval_grid=lambda ctx: sectors.reshape(-1, 1, 1),
        dtype=np.float64,
        spectral_index=False,
        to_mi=False,
        extra_kwargs={},
    )
    np.testing.assert_array_equal(result[:, 0, 0], [225.0, 315.0, 45.0, 135.0])
