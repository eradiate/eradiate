import numpy as np
import pytest

import eradiate
from eradiate import unit_registry as ureg
from eradiate.contexts import KernelContext
from eradiate.grid import PlaneParallelGridCoords
from eradiate.scenes.atmosphere import (
    HeterogeneousAtmosphere,
    MolecularAtmosphere,
    ParticleEnsemble,
)
from eradiate.scenes.geometry import PlaneParallelGeometry
from eradiate.spectral.index import SpectralIndex
from eradiate.test_tools.types import check_scene_element


def _find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def default_spectral_index(atmosphere):
    # This is a bit fragile (API stability is not guaranteed and units are not
    # checked) but it works well in both mono and ckd modes
    wavelengths = atmosphere.absorption_data._spectral_coverage.index.get_level_values(
        1
    ).values
    w = _find_nearest(wavelengths, 550.0)
    kwargs = {"w": w * ureg.nm}
    if eradiate.mode().is_ckd:
        kwargs["g"] = 0.5
    return SpectralIndex.new(**kwargs)


def test_gridded_empty(modes_all_double):
    with pytest.raises(ValueError):
        HeterogeneousAtmosphere()


@pytest.mark.parametrize(
    "geometry",
    [
        PlaneParallelGeometry(
            grid=PlaneParallelGridCoords.make_default().resampled(3, 3)
        ),
        # SphericalShellGeometry(), TBD convenience factories
    ],
    ids=[
        "grid_pparallel",
        #     "grid_spherical"
    ],
)
@pytest.mark.parametrize(
    "atm_params",
    [
        lambda: {"molecular_atmosphere": MolecularAtmosphere()},
        lambda: {
            "molecular_atmosphere": None,
            "particle_ensembles": ParticleEnsemble(),
        },
    ],
    ids=["molecular", "particle"],
)
def test_gridded_single_mono(
    mode_mono, geometry, atm_params, atmosphere_us_standard_mono
):
    atmosphere = HeterogeneousAtmosphere(geometry=geometry, **atm_params())
    kernel_context = KernelContext()
    if atmosphere.molecular_atmosphere:
        kernel_context = KernelContext(
            default_spectral_index(atmosphere.molecular_atmosphere)
        )
    check_scene_element(atmosphere, ctx=kernel_context)
