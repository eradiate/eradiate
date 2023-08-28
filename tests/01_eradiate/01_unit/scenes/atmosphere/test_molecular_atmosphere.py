"""Test cases of the _molecular_atmosphere module."""

import mitsuba as mi
import numpy as np
import numpy.testing as npt
import pytest

from eradiate import KernelContext
from eradiate import unit_registry as ureg
from eradiate.contexts import KernelContext
from eradiate.data import open_dataset
from eradiate.scenes.atmosphere import MolecularAtmosphere
from eradiate.scenes.core import Scene, traverse


def test_molecular_atmosphere_default_mono(mode_mono):
    atmosphere = MolecularAtmosphere()
    si = atmosphere.spectral_set().spectral_indices().__next__()
    template, _ = traverse(atmosphere)
    kernel_dict = template.render(KernelContext(si=si))


def test_molecular_atmosphere_default_ckd(mode_ckd):
    atmosphere = MolecularAtmosphere()
    si = atmosphere.spectral_set().spectral_indices().__next__()
    template, _ = traverse(atmosphere)
    kernel_dict = template.render(KernelContext(si=si))


def test_molecular_atmosphere_scale(mode_mono, error_handler_config):
    atmosphere = MolecularAtmosphere(
        thermoprops={
            "identifier": "afgl_1986-us_standard",
            "z": np.linspace(0, 120, 121) * ureg.km,
            "additional_molecules": False,
        },
        absorption_data=("komodo", [549.5, 550.5] * ureg.nm),
        error_handler_config=error_handler_config,
        scale=2.0,
    )
    template, _ = traverse(atmosphere)
    kernel_dict = template.render(KernelContext())
    assert kernel_dict["medium_atmosphere"]["scale"] == 2.0


def test_molecular_atmosphere_kernel_dict(mode_ckd, error_handler_config):
    """Constructor produces a valid kernel dictionary."""

    atmosphere = MolecularAtmosphere(
        thermoprops={
            "identifier": "afgl_1986-us_standard",
            "z": np.linspace(0, 80, 41) * ureg.km,
            "additional_molecules": False,
        },
        absorption_data=[
            open_dataset(path)
            for path in [
                "spectra/absorption/ckd/monotropa/monotropa-35700_35800.nc",
                "spectra/absorption/ckd/monotropa/monotropa-9600_9700.nc",
                "spectra/absorption/ckd/monotropa/monotropa-18100_18200.nc",
                "spectra/absorption/ckd/monotropa/monotropa-4700_4800.nc",
            ]
        ],
        error_handler_config=error_handler_config,
        geometry={
            "type": "spherical_shell",
            "ground_altitude": 0 * ureg.km,
            "toa_altitude": 80 * ureg.km,
        },
    )

    sig = atmosphere.spectral_set().spectral_indices()
    sis = list(sig)
    kernel_context = KernelContext(si=sis[0])

    template, params = traverse(Scene(objects={"atmosphere": atmosphere}))
    mi_scene: mi.Scene = mi.load_dict(template.render(kernel_context))

    # Mono evaluation generates valid parameter update tables
    mi_params: mi.SceneParameters = mi.traverse(mi_scene)

    # for w in eval_w:
    for si in sis:
        ctx = KernelContext(si=si)
        mi_params.update(params.render(ctx))


def test_molecular_atmosphere_switches(mode_mono, error_handler_config):
    # Absorption can be deactivated
    atmosphere = MolecularAtmosphere(
        absorption_data="spectra/absorption/mono/komodo/komodo.nc",
        has_absorption=False,
        error_handler_config=error_handler_config,
    )
    ctx = KernelContext()
    radprops = atmosphere.eval_radprops(ctx.si, optional_fields=True)
    npt.assert_allclose(radprops.sigma_a, 0.0)

    # Scattering can be deactivated
    atmosphere = MolecularAtmosphere(
        absorption_data="spectra/absorption/mono/komodo/komodo.nc",
        thermoprops={
            "identifier": "afgl_1986-us_standard",
            "z": np.linspace(0.0, 120.0, 121) * ureg.km,
            "additional_molecules": False,
        },
        has_scattering=False,
        error_handler_config=error_handler_config,
    )

    si = atmosphere.spectral_set().spectral_indices().__next__()
    radprops = atmosphere.eval_radprops(si, optional_fields=True)
    npt.assert_allclose(radprops.sigma_s, 0.0)

    # At least one must be active
    with pytest.raises(ValueError):
        MolecularAtmosphere(
            absorption_data="spectra/absorption/mono/komodo/komodo.nc",
            has_absorption=False,
            has_scattering=False,
            error_handler_config=error_handler_config,
        )
