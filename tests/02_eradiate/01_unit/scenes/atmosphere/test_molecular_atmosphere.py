"""Test cases of the _molecular_atmosphere module."""

import mitsuba as mi
import numpy.testing as npt
import pytest

from eradiate import KernelContext
from eradiate import unit_registry as ureg
from eradiate.data import data_store
from eradiate.scenes.atmosphere import MolecularAtmosphere
from eradiate.scenes.core import Scene, traverse
from eradiate.test_tools.types import check_scene_element


def test_molecular_atmosphere_default(mode_mono):
    """Default  constructor produces a valid kernel dictionary."""
    atmosphere = MolecularAtmosphere()
    check_scene_element(atmosphere)


def test_molecular_atmosphere_scale(mode_mono):
    atmosphere = MolecularAtmosphere(scale=2.0)
    template, _ = traverse(atmosphere)
    kernel_dict = template.render(KernelContext())
    assert kernel_dict["medium_atmosphere"]["scale"] == 2.0


def test_molecular_atmosphere_afgl_1986(mode_ckd):
    # afgl_1986() constructor produces a valid kernel dictionary
    atmosphere = MolecularAtmosphere.afgl_1986()
    template, params = traverse(Scene(objects={"atmosphere": atmosphere}))
    mi_scene: mi.Scene = mi.load_dict(template.render(KernelContext()))

    # CKD evaluation generates valid parameter update tables
    mi_params: mi.SceneParameters = mi.traverse(mi_scene)

    for w in [280, 550, 1040, 2120, 2400] * ureg.nm:
        ctx = KernelContext(si={"w": w, "g": 0.3})
        mi_params.update(params.render(ctx))


@pytest.fixture
def ussa76_approx_test_absorption_data_set():
    return data_store.fetch("tests/spectra/absorption/us76_u86_4-spectra-4000_25711.nc")


def test_molecular_atmosphere_ussa_1976(
    mode_mono, ussa76_approx_test_absorption_data_set
):
    # ussa_1976() constructor produces a valid kernel dictionary
    atmosphere = MolecularAtmosphere.ussa_1976(
        absorption_dataset=ussa76_approx_test_absorption_data_set,
    )
    template, params = traverse(Scene(objects={"atmosphere": atmosphere}))
    mi_scene: mi.Scene = mi.load_dict(template.render(KernelContext()))

    # CKD evaluation generates valid parameter update tables
    mi_params: mi.SceneParameters = mi.traverse(mi_scene)

    for w in [400.0, 550.0, 1040.0, 2120.0, 2400.0] * ureg.nm:
        # Note: Covered range depends on data
        #       (as of 24-01-2023, 4000-25000 cm-1, i.e. 400-2500 nm)
        ctx = KernelContext(si={"w": w})
        mi_params.update(params.render(ctx))


def test_molecular_atmosphere_switches(mode_mono):
    # Absorption can be deactivated
    atmosphere = MolecularAtmosphere(has_absorption=False)
    ctx = KernelContext()
    radprops = atmosphere.eval_radprops(ctx.si, optional_fields=True)
    npt.assert_allclose(radprops.sigma_a, 0.0)

    # Scattering can be deactivated
    atmosphere = MolecularAtmosphere(has_scattering=False)
    ctx = KernelContext()
    radprops = atmosphere.eval_radprops(ctx.si, optional_fields=True)
    npt.assert_allclose(radprops.sigma_s, 0.0)

    # At least one must be active
    with pytest.raises(ValueError):
        MolecularAtmosphere(has_absorption=False, has_scattering=False)
