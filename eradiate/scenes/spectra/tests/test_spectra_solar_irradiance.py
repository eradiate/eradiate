import numpy as np
import pytest

from eradiate import unit_registry as ureg
from eradiate._util import onedict_value
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


def test_solar_irradiance_eval(mode_mono):
    # Irradiance is correctly interpolated in mono mode
    s = SolarIrradianceSpectrum(dataset="thuillier_2003")
    spectral_ctx = SpectralContext.new(wavelength=550.0)
    assert np.allclose(s.eval(spectral_ctx), ureg.Quantity(1.87938, "W/m^2/nm"))
