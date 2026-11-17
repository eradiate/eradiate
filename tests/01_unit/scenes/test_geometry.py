import numpy as np
import pytest

from eradiate.grid import CartesianGrid
from eradiate.scenes.geometry import PlaneParallelGeometry, SphericalShellGeometry
from eradiate.units import unit_context_config as ucc
from eradiate.units import unit_context_kernel as uck
from eradiate.units import unit_registry as ureg


@pytest.mark.parametrize("cls", [PlaneParallelGeometry, SphericalShellGeometry])
def test_atmosphere_volume_to_world_units(mode_mono, cls):
    # The transform is expressed in kernel units and does not depend on config
    # units
    geometry = cls()

    def to_world(config_length, kernel_length):
        with ucc.override(length=config_length), uck.override(length=kernel_length):
            return np.array(geometry.atmosphere_volume_to_world.matrix)

    reference = to_world("m", "m")
    np.testing.assert_allclose(to_world("km", "m"), reference)
    # Linear and translation parts scale with kernel length units
    np.testing.assert_allclose(1e3 * to_world("m", "km")[:3, :], reference[:3, :])


def test_plane_parallel_crop_to_grid(mode_mono):
    grid = CartesianGrid(
        x_edges=[-1.0, 1.0] * ureg.km,
        y_edges=[-2.0, 2.0] * ureg.km,
        z_edges=[1.0, 2.0, 3.0] * ureg.km,
    )
    width = 100.0 * ureg.km

    # Only z is cropped: full width horizontally, grid extent vertically
    bbox = PlaneParallelGeometry(grid=grid, width=width, crop_to_grid="z").bbox
    np.testing.assert_allclose(bbox.min.m_as("km"), [-50.0, -50.0, 1.0])
    np.testing.assert_allclose(bbox.max.m_as("km"), [50.0, 50.0, 3.0])

    bbox = PlaneParallelGeometry(grid=grid, width=width, crop_to_grid="xyz").bbox
    np.testing.assert_allclose(bbox.min.m_as("km"), [-1.0, -2.0, 1.0])
    np.testing.assert_allclose(bbox.max.m_as("km"), [1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        PlaneParallelGeometry(crop_to_grid="zx")
