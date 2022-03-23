import attr
import numpy as np
import pytest

from eradiate import unit_registry as ureg
from eradiate.contexts import KernelDictContext
from eradiate.scenes.atmosphere import MolecularAtmosphere, PlaneParallelGeometry


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "has_absorption": True,
            "has_scattering": True,
        },
        {
            "has_absorption": True,
            "has_scattering": False,
        },
        {
            "has_absorption": False,
            "has_scattering": True,
        },
    ],
)
def test_kernel_width_plane_parallel(mode_ckd, kwargs):
    """Atmosphere has valid/correct width value in plane parallel geometry."""
    # width is AUTO
    atmosphere = MolecularAtmosphere.afgl_1986(**kwargs)
    atmosphere = attr.evolve(
        atmosphere,
        geometry=PlaneParallelGeometry(),
    )
    width = atmosphere.kernel_width_plane_parallel(ctx=KernelDictContext())
    assert width.magnitude > 0.0 and width.magnitude < np.inf  # width value is valid

    # width is set
    width_preset = 1e3 * ureg.km
    atmosphere = MolecularAtmosphere.afgl_1986(**kwargs)
    atmosphere = attr.evolve(
        atmosphere,
        geometry=PlaneParallelGeometry(width=width_preset),
    )
    width = atmosphere.kernel_width_plane_parallel(ctx=KernelDictContext())
    assert width == width_preset  # width value is correct
