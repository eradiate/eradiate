"""Test cases of the _molecular module."""

import mitsuba as mi
import numpy as np
import numpy.testing as npt
import pytest

from eradiate import unit_registry as ureg
from eradiate.contexts import KernelContext
from eradiate.scenes.atmosphere import MolecularAtmosphere
from eradiate.scenes.core import Scene, traverse
from eradiate.spectral import CKDSpectralIndex


def test_molecular_atmosphere_default_mono(mode_mono):
    atmosphere = MolecularAtmosphere()
    si = atmosphere.spectral_set().spectral_indices().__next__()
    template, _ = traverse(atmosphere)
    assert template.render(KernelContext(si=si))


def test_molecular_atmosphere_default_ckd(mode_ckd):
    atmosphere = MolecularAtmosphere()
    si = next(atmosphere.spectral_set().spectral_indices())
    template, _ = traverse(atmosphere)
    assert template.render(KernelContext(si=si))


def test_molecular_atmosphere_scale(
    mode_mono, absorption_database_error_handler_config
):
    atmosphere = MolecularAtmosphere(
        thermoprops={
            "identifier": "afgl_1986-us_standard",
            "z": np.linspace(0, 120, 121) * ureg.km,
            "additional_molecules": False,
        },
        absorption_data="komodo",
        error_handler_config=absorption_database_error_handler_config,
        scale=2.0,
    )
    template, _ = traverse(atmosphere)
    kernel_dict = template.render(KernelContext())
    assert kernel_dict["medium_atmosphere"]["scale"] == 2.0


def test_molecular_atmosphere_kernel_dict(
    mode_ckd, absorption_database_error_handler_config
):
    """Constructor produces a valid kernel dictionary."""

    atmosphere = MolecularAtmosphere(
        thermoprops={
            "identifier": "afgl_1986-us_standard",
            "z": np.linspace(0, 80, 41) * ureg.km,
            "additional_molecules": False,
        },
        absorption_data="monotropa",
        error_handler_config=absorption_database_error_handler_config,
        geometry={
            "type": "spherical_shell",
            "ground_altitude": 0 * ureg.km,
            "toa_altitude": 80 * ureg.km,
        },
    )

    sis = [CKDSpectralIndex(w=550.0, g=g) for g in [0.25, 0.75]]
    kernel_context = KernelContext(si=sis[0])

    template, params = traverse(Scene(objects={"atmosphere": atmosphere}))
    mi_scene: mi.Scene = mi.load_dict(template.render(kernel_context))

    # Mono evaluation generates valid parameter update tables
    mi_params: mi.SceneParameters = mi.traverse(mi_scene)

    # for w in eval_w:
    for si in sis:
        ctx = KernelContext(si=si)
        mi_params.update(params.render(ctx))


def test_molecular_atmosphere_switches(
    mode_mono, absorption_database_error_handler_config
):
    # Absorption can be deactivated
    atmosphere = MolecularAtmosphere(
        absorption_data="komodo",
        has_absorption=False,
        error_handler_config=absorption_database_error_handler_config,
    )
    ctx = KernelContext()
    radprops = atmosphere.eval_radprops(ctx.si, optional_fields=True)
    npt.assert_allclose(radprops.sigma_a, 0.0)

    # Scattering can be deactivated
    atmosphere = MolecularAtmosphere(
        absorption_data="komodo",
        thermoprops={
            "identifier": "afgl_1986-us_standard",
            "z": np.linspace(0.0, 120.0, 121) * ureg.km,
            "additional_molecules": False,
        },
        has_scattering=False,
        error_handler_config=absorption_database_error_handler_config,
    )

    si = next(atmosphere.spectral_set().spectral_indices())
    radprops = atmosphere.eval_radprops(si, optional_fields=True)
    npt.assert_allclose(radprops.sigma_s, 0.0)

    # At least one must be active
    with pytest.raises(ValueError):
        MolecularAtmosphere(
            absorption_data="komodo",
            has_absorption=False,
            has_scattering=False,
            error_handler_config=absorption_database_error_handler_config,
        )
