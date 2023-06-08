"""Test cases of the _molecular_atmosphere module."""

import mitsuba as mi
import numpy.testing as npt
import pytest

from eradiate import KernelContext
from eradiate import unit_registry as ureg
from eradiate.scenes.atmosphere import MolecularAtmosphere
from eradiate.scenes.core import Scene, traverse
from eradiate.test_tools.types import check_scene_element
from eradiate.test_tools.util import skipif_data_not_found


@pytest.fixture
def absorption_dataset_550nm():
    return "spectra/absorption/us76_u86_4/us76_u86_4-spectra-18000_19000.nc"


def test_molecular_atmosphere_default(mode_mono):
    """Default  constructor produces a valid kernel dictionary."""
    skipif_data_not_found(
        f"spectra/absorption/us76_u86_4/us76_u86_4-spectra-18000_19000.nc"
    )
    atmosphere = MolecularAtmosphere()
    check_scene_element(atmosphere)


def test_molecular_atmosphere_scale(mode_mono):
    skipif_data_not_found(
        f"spectra/absorption/us76_u86_4/us76_u86_4-spectra-18000_19000.nc"
    )
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


@pytest.mark.slow
def test_molecular_atmosphere_ussa_1976(mode_mono):
    # ussa_1976() constructor produces a valid kernel dictionary

    for wavenumber_range in [
        "15000_16000",
        "18000_19000",
        "22000_23000",
    ]:
        skipif_data_not_found(
            f"spectra/absorption/us76_u86_4/us76_u86_4-spectra-{wavenumber_range}.nc"
        )

    eval_w = [440.0, 550.0, 660.0] * ureg.nm
    atmosphere = MolecularAtmosphere.ussa_1976(
        wavelength_range=eval_w[
            (0, -1),
        ]
    )
    template, params = traverse(Scene(objects={"atmosphere": atmosphere}))
    mi_scene: mi.Scene = mi.load_dict(template.render(KernelContext()))

    # Mono evaluation generates valid parameter update tables
    mi_params: mi.SceneParameters = mi.traverse(mi_scene)

    for w in eval_w:
        ctx = KernelContext(si={"w": w})
        mi_params.update(params.render(ctx))


def test_molecular_atmosphere_switches(mode_mono, absorption_dataset_550nm):
    # Absorption can be deactivated
    atmosphere = MolecularAtmosphere(
        absorption_dataset=absorption_dataset_550nm,
        has_absorption=False,
    )
    ctx = KernelContext()
    radprops = atmosphere.eval_radprops(ctx.si, optional_fields=True)
    npt.assert_allclose(radprops.sigma_a, 0.0)

    # Scattering can be deactivated
    skipif_data_not_found(
        f"spectra/absorption/us76_u86_4/us76_u86_4-spectra-18000_19000.nc"
    )
    atmosphere = MolecularAtmosphere(
        absorption_dataset=absorption_dataset_550nm,
        has_scattering=False,
    )
    ctx = KernelContext()
    radprops = atmosphere.eval_radprops(ctx.si, optional_fields=True)
    npt.assert_allclose(radprops.sigma_s, 0.0)

    # At least one must be active
    with pytest.raises(ValueError):
        MolecularAtmosphere(
            absorption_dataset=absorption_dataset_550nm,
            has_absorption=False,
            has_scattering=False,
        )
