"""Test cases of the _molecular module."""

import mitsuba as mi
import numpy as np
import numpy.testing as npt
import pint
import pytest

import eradiate
import eradiate.radprops._atmosphere as atmosphere_module
from eradiate import unit_registry as ureg
from eradiate.contexts import KernelContext
from eradiate.exceptions import DataError
from eradiate.scenes.atmosphere import MolecularAtmosphere
from eradiate.scenes.core import Scene, traverse
from eradiate.spectral import CKDSpectralIndex, SpectralIndex


def _find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def _default_spectral_index(atmosphere):
    # This is a bit fragile (API stability is not guaranteed and units are not
    # checked) but it works well in both mono and ckd modes
    wavelengths = atmosphere.absorption_data._spectral_coverage.index.get_level_values(
        1
    ).values
    w = _find_nearest(wavelengths, 550.0)
    kwargs = {"w": w * ureg.nm}
    if eradiate.get_mode().is_ckd:
        kwargs["g"] = 0.5
    return SpectralIndex.new(**kwargs)


def test_molecular_atmosphere_default_mono(modes_all_unpolarized_double):
    atmosphere = MolecularAtmosphere()
    si = _default_spectral_index(atmosphere)
    template, _ = traverse(atmosphere)
    assert template.render(KernelContext(si=si))


def test_molecular_atmosphere_absorption_data_param_error(modes_all_unpolarized_double):
    # Must not raise
    MolecularAtmosphere()

    # Must raise with a comprehensive message
    with pytest.raises(DataError, match="Could not load molecular absorption database"):
        MolecularAtmosphere(absorption_data="doesnt_exist")


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
    si = _default_spectral_index(atmosphere)
    kernel_dict = template.render(KernelContext(si=si))
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

    si = _default_spectral_index(atmosphere)
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


def test_molecular_atmosphere_depolarization(mode_ckd):
    atmosphere = MolecularAtmosphere(rayleigh_depolarization=0.5)
    si = _default_spectral_index(atmosphere)
    depol = atmosphere.eval_depolarization_factor(si)
    template, _ = traverse(atmosphere)
    assert template.render(KernelContext(si=si))
    assert isinstance(depol, pint.Quantity)
    assert len(depol) == 1

    atmosphere = MolecularAtmosphere(rayleigh_depolarization=[0.1, 0.3, 0.6])
    si = _default_spectral_index(atmosphere)
    depol = atmosphere.eval_depolarization_factor(si)
    template, _ = traverse(atmosphere)
    assert template.render(KernelContext(si=si))
    assert isinstance(depol, pint.Quantity)
    assert len(depol) == 3

    atmosphere = MolecularAtmosphere(rayleigh_depolarization="bates")
    si = _default_spectral_index(atmosphere)
    depol = atmosphere.eval_depolarization_factor(si)
    template, _ = traverse(atmosphere)
    assert template.render(KernelContext(si=si))
    assert isinstance(depol, pint.Quantity)
    assert len(depol) == 1

    atmosphere = MolecularAtmosphere(rayleigh_depolarization="bodhaine")
    si = _default_spectral_index(atmosphere)
    depol = atmosphere.eval_depolarization_factor(si)
    template, _ = traverse(atmosphere)
    assert template.render(KernelContext(si=si))
    assert isinstance(depol, pint.Quantity)
    assert len(depol) == atmosphere.geometry.zgrid.n_layers


def test_molecular_atmosphere_eval_count(modes_all_unpolarized_double, monkeypatch):
    # Rendering the parameter map for one spectral index queries the absorption
    # database and evaluates Rayleigh scattering once each
    atmosphere = MolecularAtmosphere()
    si = _default_spectral_index(atmosphere)
    _, params = traverse(atmosphere)

    counts = {"sigma_a": 0, "sigma_s": 0}

    def spy(name, func):
        def wrapped(*args, **kwargs):
            counts[name] += 1
            return func(*args, **kwargs)

        return wrapped

    method = "eval_sigma_a_ckd" if eradiate.get_mode().is_ckd else "eval_sigma_a_mono"
    cls = type(atmosphere.absorption_data)
    monkeypatch.setattr(cls, method, spy("sigma_a", getattr(cls, method)))
    monkeypatch.setattr(
        atmosphere_module,
        "compute_sigma_s_air",
        spy("sigma_s", atmosphere_module.compute_sigma_s_air),
    )

    params.render(KernelContext(si=si))
    assert counts == {"sigma_a": 1, "sigma_s": 1}


def test_molecular_atmosphere_eval_count_g_points(mode_ckd, monkeypatch):
    # g-independent quantities are evaluated once for consecutive g-points
    # sharing the same wavelength
    atmosphere = MolecularAtmosphere(rayleigh_depolarization="bodhaine")
    w = _default_spectral_index(atmosphere).w
    counts = {"compute_sigma_s_air": 0, "depolarization_bodhaine": 0}

    for name in counts:
        func = getattr(atmosphere_module, name)

        def wrapped(*args, name=name, func=func, **kwargs):
            counts[name] += 1
            return func(*args, **kwargs)

        monkeypatch.setattr(atmosphere_module, name, wrapped)

    for g in [0.25, 0.75]:
        si = CKDSpectralIndex(w=w, g=g)
        atmosphere.eval_sigma_s(si)
        atmosphere.eval_depolarization_factor(si)

    assert counts == {"compute_sigma_s_air": 1, "depolarization_bodhaine": 1}
