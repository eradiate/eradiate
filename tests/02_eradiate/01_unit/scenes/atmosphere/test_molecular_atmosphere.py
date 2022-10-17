"""Test cases of the _molecular_atmosphere module."""

import numpy as np
import pytest

import eradiate
from eradiate.contexts import KernelDictContext
from eradiate.scenes.atmosphere import MolecularAtmosphere


def test_molecular_atmosphere_default(
    mode_mono, tmpdir, ussa76_approx_test_absorption_data_set
):
    """Default  constructor produces a valid kernel dictionary."""
    ctx = KernelDictContext(spectral_ctx={"wavelength": 550.0})
    atmosphere = MolecularAtmosphere(
        geometry="plane_parallel",
        absorption_data_sets=dict(us76_u86_4=ussa76_approx_test_absorption_data_set),
    )
    assert atmosphere.kernel_dict(ctx).load()


def test_molecular_atmosphere_scale(mode_mono):
    ctx = KernelDictContext()
    d = MolecularAtmosphere(geometry="plane_parallel", scale=2.0).kernel_dict(ctx)
    assert d["medium_atmosphere"]["scale"] == 2.0
    assert d.load()


@pytest.fixture
def afgl_1986_test_absorption_data_sets():
    return {
        name: eradiate.data.data_store.fetch(filename)
        for (name, filename) in [
            ("CH4", "tests/spectra/absorption/CH4-spectra-4000_11502.nc"),
            ("CO2", "tests/spectra/absorption/CO2-spectra-4000_14076.nc"),
            ("CO", "tests/spectra/absorption/CO-spectra-4000_14478.nc"),
            ("H2O", "tests/spectra/absorption/H2O-spectra-4000_25711.nc"),
            ("N2O", "tests/spectra/absorption/N2O-spectra-4000_10364.nc"),
            ("O2", "tests/spectra/absorption/O2-spectra-4000_17273.nc"),
            ("O3", "tests/spectra/absorption/O3-spectra-4000_6997.nc"),
        ]
    }


@pytest.mark.parametrize("bin", ["280", "550", "1040", "2120", "2400"])
def test_molecular_atmosphere_afgl_1986(mode_ckd, bin):
    """MolecularAtmosphere 'afgl_1986' constructor produces a valid kernel
    dictionary."""
    bin = eradiate.scenes.measure.MeasureSpectralConfig.new(bins=bin).bins[0]
    bindex = eradiate.ckd.Bindex(bin=bin, index=3)
    ctx = KernelDictContext(spectral_ctx={"bindex": bindex, "bin_set": "10nm"})
    atmosphere = MolecularAtmosphere.afgl_1986(geometry="plane_parallel")
    assert atmosphere.kernel_dict(ctx).load()


@pytest.fixture
def ussa76_approx_test_absorption_data_set():
    return eradiate.data.data_store.fetch(
        "tests/spectra/absorption/us76_u86_4-spectra-4000_25711.nc"
    )


def test_molecular_atmosphere_ussa_1976(
    mode_mono,
    tmpdir,
    ussa76_approx_test_absorption_data_set,
):
    """MolecularAtmosphere 'ussa_1976' constructor produces a valid kernel
    dictionary."""
    ctx = KernelDictContext(spectral_ctx={"wavelength": 550.0})
    atmosphere = MolecularAtmosphere.ussa_1976(
        geometry="plane_parallel",
        absorption_data_sets=dict(us76_u86_4=ussa76_approx_test_absorption_data_set),
    )
    assert atmosphere.kernel_dict(ctx).load()


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
