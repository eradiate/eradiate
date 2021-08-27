import numpy as np
import pytest

import eradiate
from eradiate import unit_registry as ureg
from eradiate._mode import ModeFlags
from eradiate._util import onedict_value
from eradiate.ckd import BinSet
from eradiate.contexts import KernelDictContext, SpectralContext
from eradiate.scenes.spectra import SolarIrradianceSpectrum


def test_solar_irradiance(mode_mono):
    from mitsuba.core.xml import load_dict

    # Default context
    ctx = KernelDictContext()

    # We can instantiate the element
    s = SolarIrradianceSpectrum()

    # Unsupported solar spectrum keywords raise
    with pytest.raises(ValueError):
        SolarIrradianceSpectrum(dataset="doesnt_exist")

    # Produced kernel dict is valid
    assert load_dict(onedict_value(s.kernel_dict(ctx))) is not None

    # A more detailed specification still produces a valid object
    s = SolarIrradianceSpectrum(scale=2.0)
    assert load_dict(onedict_value(s.kernel_dict(ctx))) is not None

    # Element doesn't work out of the supported spectral range
    s = SolarIrradianceSpectrum(dataset="thuillier_2003")

    with pytest.raises(ValueError):
        ctx = KernelDictContext(spectral_ctx={"wavelength": 2400.0})
        s.kernel_dict(ctx)

    # solid_2017_mean dataset can be used
    ctx = KernelDictContext()
    s = SolarIrradianceSpectrum(dataset="solid_2017_mean")
    assert load_dict(onedict_value(s.kernel_dict(ctx)))


def test_solar_irradiance_eval(modes_all):
    # Irradiance is correctly interpolated in mono mode
    s = SolarIrradianceSpectrum(dataset="thuillier_2003")

    if eradiate.mode().has_flags(ModeFlags.ANY_MONO):
        spectral_ctx = SpectralContext.new(wavelength=550.0)
        # Reference value computed manually
        assert np.allclose(s.eval(spectral_ctx), ureg.Quantity(1.87938, "W/m^2/nm"))

    elif eradiate.mode().has_flags(ModeFlags.ANY_CKD):
        bin_set = BinSet.from_db("10nm_test")
        bin = bin_set.select_bins("550")[0]
        bindex = bin.bindexes[0]
        spectral_ctx = SpectralContext.new(bindex=bindex)
        # Reference value computed manually
        assert np.allclose(s.eval(spectral_ctx), ureg.Quantity(1.871527, "W/m^2/nm"))

    else:
        assert False
