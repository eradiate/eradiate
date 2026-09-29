import numpy as np
import pytest

from eradiate.scenes.geometry import PlaneParallelGeometry, SphericalShellGeometry
from eradiate.units import unit_context_config as ucc
from eradiate.units import unit_context_kernel as uck


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
