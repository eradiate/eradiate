import numpy as np
import pytest

import eradiate
from eradiate import unit_registry as ureg
from eradiate.ckd import BinSet
from eradiate.contexts import KernelDictContext, SpectralContext
from eradiate.exceptions import DataError
from eradiate.scenes.core import KernelDict
from eradiate.scenes.spectra import SolarIrradianceSpectrum


def test_solar_irradiance(mode_mono):

    # Default context
    ctx = KernelDictContext()

    # We can instantiate the element
    s = SolarIrradianceSpectrum()

    # Unsupported solar spectrum keywords raise
    with pytest.raises(DataError):
        SolarIrradianceSpectrum(dataset="doesnt_exist")

    # Produced kernel dict is valid
    assert KernelDict.from_elements(s, ctx=ctx).load()

    # A more detailed specification still produces a valid object
    s = SolarIrradianceSpectrum(scale=2.0)
    assert KernelDict.from_elements(s, ctx=ctx).load()

    # Element doesn't work out of the supported spectral range
    s = SolarIrradianceSpectrum(dataset="thuillier_2003")

    with pytest.raises(ValueError):
        ctx = KernelDictContext(spectral_ctx={"wavelength": 2400.0})
        s.kernel_dict(ctx)

    # solid_2017_mean dataset can be used
    ctx = KernelDictContext()
    s = SolarIrradianceSpectrum(dataset="solid_2017_mean")
    assert KernelDict.from_elements(s, ctx=ctx).load()


def test_solar_irradiance_eval(modes_all):
    # Irradiance is correctly interpolated in mono mode
    s = SolarIrradianceSpectrum(dataset="thuillier_2003")

    if eradiate.mode().is_mono:
        spectral_ctx = SpectralContext.new(wavelength=550.0)
        # Reference value computed manually
        assert np.allclose(s.eval(spectral_ctx), ureg.Quantity(1.87938, "W/m^2/nm"))

    elif eradiate.mode().is_ckd:
        bin_set = BinSet.from_db("10nm")
        bin = bin_set.select_bins("550")[0]
        bindex = bin.bindexes[0]
        spectral_ctx = SpectralContext.new(bindex=bindex)
        # Reference value computed manually
        assert np.allclose(s.eval(spectral_ctx), ureg.Quantity(1.871527, "W/m^2/nm"))

    else:
        assert False


def test_solar_irradiance_scale(mode_mono):
    s = SolarIrradianceSpectrum(dataset="thuillier_2003")

    # We can scale the spectrum using a float
    s_scaled_float = SolarIrradianceSpectrum(dataset="thuillier_2003", scale=10.0)
    assert (
        s_scaled_float.eval_mono(550.0 * ureg.nm) == s.eval_mono(550.0 * ureg.nm) * 10.0
    )


def test_solar_irradiance_datetime(mode_mono):
    s = SolarIrradianceSpectrum(dataset="thuillier_2003")

    # We can also use a datetime to scale the spectrum
    s_scaled_datetime = SolarIrradianceSpectrum(
        dataset="thuillier_2003", datetime="2021-11-18"
    )
    assert np.isclose(
        s_scaled_datetime.eval_mono(550.0 * ureg.nm),
        s.eval_mono(550.0 * ureg.nm) * 0.98854537**2,
    )
