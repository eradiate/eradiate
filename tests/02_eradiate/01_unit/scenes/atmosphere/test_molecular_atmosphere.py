"""Test cases of the _molecular_atmosphere module."""

import mitsuba as mi
import numpy as np
import pytest

from eradiate import unit_registry as ureg
from eradiate.ckd import Bindex
from eradiate.contexts import KernelDictContext
from eradiate.data import data_store
from eradiate.scenes.atmosphere import MolecularAtmosphere
from eradiate.scenes.core import Scene, traverse
from eradiate.scenes.measure import MeasureSpectralConfig
from eradiate.test_tools.types import check_scene_element


def test_molecular_atmosphere_default(mode_mono):
    """Default  constructor produces a valid kernel dictionary."""
    atmosphere = MolecularAtmosphere()
    check_scene_element(atmosphere)


def test_molecular_atmosphere_scale(mode_mono):
    atmosphere = MolecularAtmosphere(scale=2.0)
    template, params = traverse(atmosphere)
    kernel_dict = template.render(KernelDictContext())
    assert kernel_dict["medium_atmosphere"]["scale"] == 2.0


def test_molecular_atmosphere_afgl_1986(mode_ckd):
    # afgl_1986() constructor produces a valid kernel dictionary
    atmosphere = MolecularAtmosphere.afgl_1986()
    template, params = traverse(Scene(objects={"atmosphere": atmosphere}))
    mi_scene: mi.Scene = mi.load_dict(template.render(KernelDictContext()))

    # CKD evaluation generates valid parameter update tables
    mi_params: mi.SceneParameters = mi.traverse(mi_scene)

    for ckd_bin in ["280", "550", "1040", "2120", "2400"]:
        ctx = KernelDictContext(
            spectral_ctx={
                "bindex": Bindex(
                    bin=MeasureSpectralConfig.new(bins=ckd_bin).bins[0], index=3
                ),
                "bin_set": "10nm",
            }
        )
        mi_params.update(params.render(ctx))


@pytest.fixture
def ussa76_approx_test_absorption_data_set():
    return data_store.fetch("tests/spectra/absorption/us76_u86_4-spectra-4000_25711.nc")


def test_molecular_atmosphere_ussa_1976(
    mode_mono, ussa76_approx_test_absorption_data_set
):
    # ussa_1976() constructor produces a valid kernel dictionary
    atmosphere = MolecularAtmosphere.ussa_1976(
        absorption_data_sets={"us76_u86_4": ussa76_approx_test_absorption_data_set},
    )
    template, params = traverse(Scene(objects={"atmosphere": atmosphere}))
    mi_scene: mi.Scene = mi.load_dict(template.render(KernelDictContext()))

    # CKD evaluation generates valid parameter update tables
    mi_params: mi.SceneParameters = mi.traverse(mi_scene)

    for w in [280.0, 550.0, 1040.0, 2120.0, 2400.0] * ureg.nm:
        ctx = KernelDictContext(spectral_ctx={"wavelength": w})
        mi_params.update(params.render(ctx))


def test_molecular_atmosphere_switches(mode_mono):
    # Absorption can be deactivated
    atmosphere = MolecularAtmosphere(has_absorption=False)
    ctx = KernelDictContext()
    assert np.all(
        atmosphere.eval_radprops(ctx.spectral_ctx, optional_fields=True).sigma_a == 0.0
    )

    # Scattering can be deactivated
    atmosphere = MolecularAtmosphere(has_scattering=False)
    ctx = KernelDictContext()
    assert np.all(
        atmosphere.eval_radprops(ctx.spectral_ctx, optional_fields=True).sigma_s == 0.0
    )

    # At least one must be active
    with pytest.raises(ValueError):
        MolecularAtmosphere(has_absorption=False, has_scattering=False)
