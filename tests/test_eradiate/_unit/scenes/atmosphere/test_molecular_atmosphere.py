"""Test cases of the _molecular_atmosphere module."""

import numpy as np
import pytest

from eradiate import path_resolver
from eradiate.contexts import KernelDictContext, SpectralContext
from eradiate.scenes.atmosphere import MolecularAtmosphere
from eradiate.scenes.core import KernelDict


def test_molecular_atmosphere_default(
    mode_mono, tmpdir, ussa76_approx_test_absorption_data_set
):
    """Default MolecularAtmosphere constructor produces a valid kernel
    dictionary."""
    spectral_ctx = SpectralContext.new(wavelength=550.0)
    ctx = KernelDictContext(spectral_ctx=spectral_ctx)
    atmosphere = MolecularAtmosphere(
        absorption_data_sets=dict(us76_u86_4=ussa76_approx_test_absorption_data_set)
    )
    assert KernelDict.from_elements(atmosphere, ctx=ctx).load() is not None


def test_molecular_atmosphere_scale(mode_mono):
    ctx = KernelDictContext()
    d = MolecularAtmosphere(scale=2.0).kernel_dict(ctx)
    assert d["medium_atmosphere"]["scale"] == 2.0
    assert d.load()


@pytest.fixture
def afgl_1986_test_absorption_data_sets():
    return {
        "CH4": path_resolver.resolve(
            "tests/spectra/absorption/CH4-spectra-4000_11502.nc"
        ),
        "CO2": path_resolver.resolve(
            "tests/spectra/absorption/CO2-spectra-4000_14076.nc"
        ),
        "CO": path_resolver.resolve(
            "tests/spectra/absorption/CO-spectra-4000_14478.nc"
        ),
        "H2O": path_resolver.resolve(
            "tests/spectra/absorption/H2O-spectra-4000_25711.nc"
        ),
        "N2O": path_resolver.resolve(
            "tests/spectra/absorption/N2O-spectra-4000_10364.nc"
        ),
        "O2": path_resolver.resolve(
            "tests/spectra/absorption/O2-spectra-4000_17273.nc"
        ),
        "O3": path_resolver.resolve("tests/spectra/absorption/O3-spectra-4000_6997.nc"),
    }


def test_molecular_atmosphere_afgl_1986(
    mode_mono,
    tmpdir,
    afgl_1986_test_absorption_data_sets,
):
    """MolecularAtmosphere 'afgl_1986' constructor produces a valid kernel
    dictionary."""
    spectral_ctx = SpectralContext.new(wavelength=550.0)
    ctx = KernelDictContext(spectral_ctx=spectral_ctx)
    atmosphere = MolecularAtmosphere.afgl_1986(
        absorption_data_sets=afgl_1986_test_absorption_data_sets
    )
    assert KernelDict.from_elements(atmosphere, ctx=ctx).load() is not None


@pytest.fixture
def ussa76_approx_test_absorption_data_set():
    return path_resolver.resolve(
        "tests/spectra/absorption/us76_u86_4-spectra-4000_25711.nc"
    )


def test_molecular_atmosphere_ussa1976(
    mode_mono,
    tmpdir,
    ussa76_approx_test_absorption_data_set,
):
    """MolecularAtmosphere 'ussa1976' constructor produces a valid kernel
    dictionary."""
    spectral_ctx = SpectralContext.new(wavelength=550.0)
    ctx = KernelDictContext(spectral_ctx=spectral_ctx)
    atmosphere = MolecularAtmosphere.ussa1976(
        absorption_data_sets=dict(us76_u86_4=ussa76_approx_test_absorption_data_set)
    )
    assert KernelDict.from_elements(atmosphere, ctx=ctx).load() is not None


def test_molecular_atmosphere_switches(mode_mono):
    # Absorption can be deactivated
    atmosphere = MolecularAtmosphere(has_absorption=False)
    ctx = KernelDictContext()
    assert np.all(atmosphere.eval_radprops(ctx.spectral_ctx).sigma_a == 0.0)

    # Scattering can be deactivated
    atmosphere = MolecularAtmosphere(has_scattering=False)
    ctx = KernelDictContext()
    assert np.all(atmosphere.eval_radprops(ctx.spectral_ctx).sigma_s == 0.0)

    # At least one must be active
    with pytest.raises(ValueError):
        MolecularAtmosphere(has_absorption=False, has_scattering=False)
